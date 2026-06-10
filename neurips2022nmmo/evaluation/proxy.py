import zmq
import nmmo
import cloudpickle as pickle
from typing import Dict, Any, Optional, Type

from neurips2022nmmo.timer import timer
from neurips2022nmmo.evaluation.team import Team


class Req:
    method: str
    params: dict = {}

    def __init__(self, method: str, params: dict = {}) -> None:
        self.method = method
        self.params = params


class Rsp:
    result: Any
    exception: Optional[Exception] = None

    def __init__(self,
                 result: Any,
                 exception: Optional[Exception] = None) -> None:
        self.result = result
        self.exception = exception


class ProxyTeam(Team):
    _context: zmq.Context
    _sock: zmq.Socket
    remote: str

    def __init__(
        self,
        team_id: str,
        env_config: nmmo.config.Config,
        host: str,
        port: int,
        reset_timeout=5000,
        act_timeout=600,
        **kwargs,
    ) -> None:
        super().__init__(team_id, env_config, **kwargs)

        self._context = zmq.Context()
        self._sock = self._context.socket(zmq.REQ)
        self.remote = f"tcp://{host}:{port}"
        self._sock.connect(self.remote)
        self._init_remote(team_id, env_config)
        self.reset_timeout = reset_timeout
        self.act_timeout = act_timeout

    def _init_remote(self, team_id: str, env_config: nmmo.config.Config):
        req = Req("init", {"team_id": team_id, "env_config": env_config})
        rsp = self._comm(req)
        assert rsp.result == True, "init remote failed"

    def _comm(self, req: Req) -> Rsp:
        data = pickle.dumps(req)
        self._sock.send(data)
        data = self._sock.recv()
        rsp: Rsp = pickle.loads(data)
        if rsp.exception:
            raise rsp.exception
        return rsp

    def act(self, observations: Dict[int, dict]) -> Dict[int, dict]:
        with timer.count("act"):
            req = Req("act", {"observations": observations})
            rsp = self._comm(req)
        cost = timer.costs["act"][-1] * 1000
        if cost > self.act_timeout:
            raise TimeoutError(
                f"team[{self.id}].act() took {cost:.1f}ms which exceeds {self.act_timeout}ms"
            )
        return rsp.result

    def reset(self) -> None:
        with timer.count("reset"):
            req = Req("reset")
            self._comm(req)
        cost = timer.costs["reset"][-1] * 1000
        if cost > self.reset_timeout:
            raise TimeoutError(
                f"team[{self.id}].reset() took {cost:.1f}ms which exceeds {self.reset_timeout}ms"
            )

    def ping(self) -> None:
        req = Req("ping")
        self._comm(req)

    def stop(self) -> None:
        req = Req("stop")
        self._comm(req)


class TeamServer:
    team: Team = None
    host: str
    port: int
    addr: str
    _context: zmq.Context
    _sock: zmq.Socket
    team_klass: Type[Team]
    init_params: dict

    def __init__(self, host: str, port: int, team_klass: Type,
                 init_params: dict) -> None:
        self.host = host
        self.port = port
        self.addr = f"tcp://{host}:{port}"
        self._context = zmq.Context()
        self._sock = self._context.socket(zmq.REP)
        self.team_klass = team_klass
        self.init_params = init_params

    def _recv(self) -> Req:
        data = self._sock.recv()
        req: Req = pickle.loads(data)
        return req

    def _send(self, rsp: Rsp) -> None:
        data = pickle.dumps(rsp)
        self._sock.send(data)

    def run(self) -> None:
        self._sock.bind(self.addr)

        while 1:
            req = self._recv()
            if req.method == "init":
                self.team = self.team_klass(**req.params, **self.init_params)
                rsp = Rsp(True)
            elif req.method in ["act", "reset"]:
                if self.team:
                    result = getattr(self.team, req.method)(**req.params)
                    rsp = Rsp(result)
                else:
                    rsp = Rsp(None, RuntimeError("team is not initialized"))
            elif req.method == "ping":
                rsp = Rsp("pong")
            elif req.method == "stop":
                self._send(Rsp("ok"))
                break
            else:
                rsp = Rsp(None, RuntimeError(f"invalid method[{req.method}]"))
            self._send(rsp)


if __name__ == "__main__":

    def server_run():
        from neurips2022nmmo import scripted
        server = TeamServer("127.0.0.1", 12345, scripted.RandomTeam, {})
        server.run()

    import threading
    thread = threading.Thread(target=server_run, daemon=True)
    thread.start()

    import time
    time.sleep(1)

    from neurips2022nmmo.config import CompetitionConfig
    team = ProxyTeam("random", CompetitionConfig(), "127.0.0.1", 12345)

    from neurips2022nmmo.env.team_based_env import TeamBasedEnv
    env = TeamBasedEnv(CompetitionConfig())
    observations_by_team = env.reset()

    actions = team.act(observations_by_team[0])
    print(actions)

    team.reset()
