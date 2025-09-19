from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.v0043_step_state_item import V0043StepStateItem
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.v0043_process_exit_code_verbose import V0043ProcessExitCodeVerbose
    from ..models.v0043_step_cpu import V0043StepCPU
    from ..models.v0043_step_nodes import V0043StepNodes
    from ..models.v0043_step_statistics import V0043StepStatistics
    from ..models.v0043_step_step import V0043StepStep
    from ..models.v0043_step_task import V0043StepTask
    from ..models.v0043_step_tasks import V0043StepTasks
    from ..models.v0043_step_time import V0043StepTime
    from ..models.v0043_step_tres import V0043StepTres


T = TypeVar("T", bound="V0043Step")


@_attrs_define
class V0043Step:
    time: Union[Unset, "V0043StepTime"] = UNSET
    exit_code: Union[Unset, "V0043ProcessExitCodeVerbose"] = UNSET
    nodes: Union[Unset, "V0043StepNodes"] = UNSET
    tasks: Union[Unset, "V0043StepTasks"] = UNSET
    pid: Union[Unset, str] = UNSET
    """ Deprecated; Process ID """
    cpu: Union[Unset, "V0043StepCPU"] = UNSET
    kill_request_user: Union[Unset, str] = UNSET
    """ User ID that requested termination of the step """
    state: Union[Unset, list[V0043StepStateItem]] = UNSET
    """ Current state """
    statistics: Union[Unset, "V0043StepStatistics"] = UNSET
    step: Union[Unset, "V0043StepStep"] = UNSET
    task: Union[Unset, "V0043StepTask"] = UNSET
    tres: Union[Unset, "V0043StepTres"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        time: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.time, Unset):
            time = self.time.to_dict()

        exit_code: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.exit_code, Unset):
            exit_code = self.exit_code.to_dict()

        nodes: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.nodes, Unset):
            nodes = self.nodes.to_dict()

        tasks: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.tasks, Unset):
            tasks = self.tasks.to_dict()

        pid = self.pid

        cpu: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.cpu, Unset):
            cpu = self.cpu.to_dict()

        kill_request_user = self.kill_request_user

        state: Union[Unset, list[str]] = UNSET
        if not isinstance(self.state, Unset):
            state = []
            for state_item_data in self.state:
                state_item = state_item_data.value
                state.append(state_item)

        statistics: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.statistics, Unset):
            statistics = self.statistics.to_dict()

        step: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.step, Unset):
            step = self.step.to_dict()

        task: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.task, Unset):
            task = self.task.to_dict()

        tres: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.tres, Unset):
            tres = self.tres.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if time is not UNSET:
            field_dict["time"] = time
        if exit_code is not UNSET:
            field_dict["exit_code"] = exit_code
        if nodes is not UNSET:
            field_dict["nodes"] = nodes
        if tasks is not UNSET:
            field_dict["tasks"] = tasks
        if pid is not UNSET:
            field_dict["pid"] = pid
        if cpu is not UNSET:
            field_dict["CPU"] = cpu
        if kill_request_user is not UNSET:
            field_dict["kill_request_user"] = kill_request_user
        if state is not UNSET:
            field_dict["state"] = state
        if statistics is not UNSET:
            field_dict["statistics"] = statistics
        if step is not UNSET:
            field_dict["step"] = step
        if task is not UNSET:
            field_dict["task"] = task
        if tres is not UNSET:
            field_dict["tres"] = tres

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.v0043_process_exit_code_verbose import V0043ProcessExitCodeVerbose
        from ..models.v0043_step_cpu import V0043StepCPU
        from ..models.v0043_step_nodes import V0043StepNodes
        from ..models.v0043_step_statistics import V0043StepStatistics
        from ..models.v0043_step_step import V0043StepStep
        from ..models.v0043_step_task import V0043StepTask
        from ..models.v0043_step_tasks import V0043StepTasks
        from ..models.v0043_step_time import V0043StepTime
        from ..models.v0043_step_tres import V0043StepTres

        d = dict(src_dict)
        _time = d.pop("time", UNSET)
        time: Union[Unset, V0043StepTime]
        if isinstance(_time, Unset):
            time = UNSET
        else:
            time = V0043StepTime.from_dict(_time)

        _exit_code = d.pop("exit_code", UNSET)
        exit_code: Union[Unset, V0043ProcessExitCodeVerbose]
        if isinstance(_exit_code, Unset):
            exit_code = UNSET
        else:
            exit_code = V0043ProcessExitCodeVerbose.from_dict(_exit_code)

        _nodes = d.pop("nodes", UNSET)
        nodes: Union[Unset, V0043StepNodes]
        if isinstance(_nodes, Unset):
            nodes = UNSET
        else:
            nodes = V0043StepNodes.from_dict(_nodes)

        _tasks = d.pop("tasks", UNSET)
        tasks: Union[Unset, V0043StepTasks]
        if isinstance(_tasks, Unset):
            tasks = UNSET
        else:
            tasks = V0043StepTasks.from_dict(_tasks)

        pid = d.pop("pid", UNSET)

        _cpu = d.pop("CPU", UNSET)
        cpu: Union[Unset, V0043StepCPU]
        if isinstance(_cpu, Unset):
            cpu = UNSET
        else:
            cpu = V0043StepCPU.from_dict(_cpu)

        kill_request_user = d.pop("kill_request_user", UNSET)

        state = []
        _state = d.pop("state", UNSET)
        for state_item_data in _state or []:
            state_item = V0043StepStateItem(state_item_data)

            state.append(state_item)

        _statistics = d.pop("statistics", UNSET)
        statistics: Union[Unset, V0043StepStatistics]
        if isinstance(_statistics, Unset):
            statistics = UNSET
        else:
            statistics = V0043StepStatistics.from_dict(_statistics)

        _step = d.pop("step", UNSET)
        step: Union[Unset, V0043StepStep]
        if isinstance(_step, Unset):
            step = UNSET
        else:
            step = V0043StepStep.from_dict(_step)

        _task = d.pop("task", UNSET)
        task: Union[Unset, V0043StepTask]
        if isinstance(_task, Unset):
            task = UNSET
        else:
            task = V0043StepTask.from_dict(_task)

        _tres = d.pop("tres", UNSET)
        tres: Union[Unset, V0043StepTres]
        if isinstance(_tres, Unset):
            tres = UNSET
        else:
            tres = V0043StepTres.from_dict(_tres)

        v0043_step = cls(
            time=time,
            exit_code=exit_code,
            nodes=nodes,
            tasks=tasks,
            pid=pid,
            cpu=cpu,
            kill_request_user=kill_request_user,
            state=state,
            statistics=statistics,
            step=step,
            task=task,
            tres=tres,
        )

        v0043_step.additional_properties = d
        return v0043_step

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
