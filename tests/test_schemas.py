from polar_flow.server.schemas import TaskCreate


def test_taskcreate_priority_default_is_100():
    t = TaskCreate(
        name="n",
        command="echo hi",
        requested_gpus="0",
        working_dir="/tmp",
    )
    assert t.priority == 100
