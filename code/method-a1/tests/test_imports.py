"""验证 Method-A1 包在无 OpenDSS COM 环境下可以导入。"""


def test_a1_packages_import_without_open_dss():
    """包导入不应在初始化阶段触发 OpenDSS 调用。"""
    import src
    import src.data_generation
    import src.model

    assert src is not None
    assert src.data_generation is not None
    assert src.model is not None
