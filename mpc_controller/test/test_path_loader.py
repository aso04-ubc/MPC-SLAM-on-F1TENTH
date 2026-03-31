from pathlib import Path

from mpc_controller.mpc_core.path_loader import load_path_reference_from_csv


def test_load_path_reference_from_headered_csv(tmp_path):
    csv_path = Path(tmp_path) / 'centerline.csv'
    csv_path.write_text(
        'x,y,v_ref\n'
        '0.0,0.0,1.0\n'
        '1.0,0.0,1.1\n'
        '2.0,1.0,1.2\n',
        encoding='utf-8',
    )

    path = load_path_reference_from_csv(str(csv_path), default_speed=0.8)

    assert len(path.x) == 3
    assert path.x[2] == 2.0
    assert path.y[2] == 1.0
    assert path.v_ref == [1.0, 1.1, 1.2]


def test_load_path_reference_from_headerless_csv(tmp_path):
    csv_path = Path(tmp_path) / 'centerline_plain.csv'
    csv_path.write_text(
        '0.0,0.0\n'
        '1.0,0.0\n'
        '2.0,0.5\n',
        encoding='utf-8',
    )

    path = load_path_reference_from_csv(str(csv_path), default_speed=1.3)

    assert len(path.x) == 3
    assert path.v_ref == [1.3, 1.3, 1.3]
    assert path.s[0] == 0.0
    assert path.s[-1] > 2.0
