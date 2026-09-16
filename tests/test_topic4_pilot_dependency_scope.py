from scripts.resume_topic4_multidimensional_pilot import dependency_paths


def test_closure_includes_imports_dynamic_loader_but_not_unrelated_audit(tmp_path):
    (tmp_path/'scripts').mkdir();(tmp_path/'src').mkdir()
    (tmp_path/'scripts/run.py').write_text("from src import dynamics\nloader = 'helper.py'\n")
    (tmp_path/'src/dynamics.py').write_text('from . import observation\n')
    (tmp_path/'src/observation.py').write_text('')
    (tmp_path/'scripts/helper.py').write_text('')
    (tmp_path/'scripts/unrelated_audit.py').write_text('')
    actual=dependency_paths(tmp_path,{'scripts/run.py'})
    assert actual=={'scripts/run.py','scripts/helper.py','src/dynamics.py','src/observation.py'}
