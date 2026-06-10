import nmmo
import openskill
from tqdm import tqdm
from loguru import logger
from concurrent import futures
from typing import Dict, List, Optional

from neurips2022nmmo.evaluation.rating import RatingSystem
from neurips2022nmmo.evaluation.team import Team
from neurips2022nmmo.env.team_based_env import TeamBasedEnv
from neurips2022nmmo.timer import timer
from neurips2022nmmo.evaluation import analyzer
from neurips2022nmmo.exception import TeamTimeoutError


class RollOut(object):
    env: TeamBasedEnv
    teams: List[Team]
    parallel: bool
    show_progress: bool
    _executor: Optional[futures.ThreadPoolExecutor] = None

    def __init__(
        self,
        env_config: nmmo.config.Config,
        teams: List[Team],
        parallel: bool = False,
        show_progress: bool = True,
    ) -> None:
        assert len(teams) == len(
            env_config.PLAYERS
        ), f"number of teams ({len(teams)}) is different with config ({len(env_config.PLAYERS)})"

        # use team.id as agent.name, so players can be identified in replay
        for i, team in enumerate(teams):

            class Agent(nmmo.Agent):
                name = f"{team.id}_"
                policy = f"{team.id}_"

            env_config.PLAYERS[i] = Agent

        self.env = TeamBasedEnv(env_config)
        self.teams = teams
        self.parallel = parallel
        if parallel:
            self._executor = futures.ThreadPoolExecutor(max_workers=len(teams))
        self.show_progress = show_progress

    def reset(self) -> Dict[int, Dict[int, dict]]:
        observations_by_team = self.env.reset()
        if self.parallel:
            fs: Dict[int, futures.Future] = {}
            for team_idx, team in enumerate(self.teams):
                fs[team_idx] = self._executor.submit(team.reset)
            for team_idx, f in fs.items():
                try:
                    f.result()
                except Exception as e:
                    logger.warning(e)
        else:
            for team in self.teams:
                team.reset()
                team.n_timeout = 0
        return observations_by_team

    def run(
        self,
        n_timestep: int = 1024,
        n_episode: int = 1,
        render: bool = False,
    ) -> List[Dict[int, analyzer.TeamResult]]:
        results = []
        episode = 0
        while episode < n_episode:
            episode += 1
            print(f"Start Episode {episode}!")

            if self.show_progress:
                pbar = tqdm(total=n_timestep)

            observations_by_team = self.reset()

            timestep = 0
            while observations_by_team and timestep < n_timestep:
                if self.show_progress:
                    pbar.update()

                if render:
                    self.env.render()

                with timer.count("get_actions"):
                    actions_by_team = self._get_actions_by_team(
                        observations_by_team)

                with timer.count("env.step"):
                    observations_by_team, _, _, _ = self.env.step(
                        actions_by_team)

                timestep += 1

            if self.show_progress:
                pbar.close()

            self._end_episode(episode, results)

        return self._end_run(results)

    def _print(self,
               result_by_team: Dict[int, analyzer.TeamResult],
               ratings: Optional[Dict[int, openskill.Rating]] = None,
               topn_probs: Optional[Dict[int, float]] = None,
               n: int = 1):
        def camelize(x):
            return ''.join(word.title() for word in x.split('_'))

        result_names = analyzer.TeamResult.names()

        import prettytable
        from prettytable import PrettyTable
        table = PrettyTable()
        field_names = ["Team"] + list(map(camelize, result_names))
        if ratings:
            field_names += ["Rating"]
        if topn_probs:
            field_names += [f"Top{n}Ratio"]

        table.field_names = field_names
        table.hrules = prettytable.ALL
        table.vrules = prettytable.FRAME
        for name in field_names:
            table.align[name] = "l"
        for team_idx, result in result_by_team.items():
            row = [
                f"{self.teams[team_idx].id}\n({self.teams[team_idx].policy_id})"
            ] + [f"{getattr(result, key):.2f}" for key in result_names]
            if ratings:
                row += [f"{ratings[team_idx].mu:.2f}"]
            if topn_probs:
                row += [f"{topn_probs[team_idx]:.2f}"]
            table.add_row(row)
        print(table)

    def _get_actions_by_team(
            self, observations_by_team) -> Dict[int, Dict[int, dict]]:
        actions_by_team: Dict[int, Dict[int, dict]] = {}

        if self.parallel:
            fs: Dict[int, futures.Future] = {}
            for team_idx, observations in observations_by_team.items():
                team = self.teams[team_idx]
                fs[team_idx] = self._executor.submit(team.act, observations)
            for team_idx, f in fs.items():
                try:
                    actions_by_team[team_idx] = f.result()
                except TeamTimeoutError as e:
                    logger.warning(e)
                    self.teams[team_idx].n_timeout += 1
                except Exception as e:
                    logger.warning(e)
        else:
            for team_idx, observations in observations_by_team.items():
                team = self.teams[team_idx]
                actions_by_team[team_idx] = team.act(observations)

        return actions_by_team

    def _end_episode(self, episode: int,
                     results: List[Dict[int, analyzer.TeamResult]]):
        with timer.count("env.terminal"):
            self.env.terminal()

        metrics_by_team = self.env.metrices_by_team()
        policy_id_by_team = {
            i: self.teams[i].policy_id
            for i in metrics_by_team.keys()
        }
        n_timeout_by_team = {
            i: self.teams[i].n_timeout
            for i in metrics_by_team.keys()
        }
        result = analyzer.gen_result(policy_id_by_team, metrics_by_team,
                                     n_timeout_by_team)
        results.append(result)

        print(f"Result of Episode {episode}:")
        self._print(results[-1])

    def _end_run(self, results: List[Dict[int, analyzer.TeamResult]]):
        ret = results

        rs = RatingSystem(self.teams)
        for result_by_team in results:
            rs.update(
                self.teams,
                [result.total_score for result in result_by_team.values()],
            )

        results: Dict[int, analyzer.TeamResult] = analyzer.avg_results(results)
        ratings = rs.get_team_ratings(self.teams)

        print("Final average Result:")
        self._print(
            results,
            # ratings,
            topn_probs=analyzer.topn_probs(ret),
        )

        return ret


if __name__ == "__main__":
    from neurips2022nmmo import CompetitionConfig
    config = CompetitionConfig()

    from neurips2022nmmo import scripted
    teams = []
    teams.extend(
        [scripted.FisherTeam(f"Fisher-{i}", config) for i in range(2)])
    teams.extend(
        [scripted.HerbalistTeam(f"Herbalist-{i}", config) for i in range(2)])
    teams.extend(
        [scripted.ProspectorTeam(f"Prospector-{i}", config) for i in range(2)])
    teams.extend(
        [scripted.CarverTeam(f"Carver-{i}", config) for i in range(2)])
    teams.extend(
        [scripted.AlchemistTeam(f"Alchemist-{i}", config) for i in range(2)])
    teams.extend([scripted.MeleeTeam(f"Melee-{i}", config) for i in range(2)])
    teams.extend([scripted.RangeTeam(f"Range-{i}", config) for i in range(2)])
    teams.extend([scripted.MageTeam(f"Mage-{i}", config) for i in range(2)])
    ro = RollOut(
        config,
        teams,
        show_progress=True,
    )
    ro.run(n_timestep=2000, n_episode=2)
