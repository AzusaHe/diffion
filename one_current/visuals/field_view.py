"""Prepare binned concentration fields, then embed an offline HTML field viewer."""
import base64
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "one_current" / "random"
DATA = RUN / "visuals" / "field_data.json"


def prepare():
    import numpy as np
    selected = np.flatnonzero(np.load(REPO / "data/train_split.npy") == "id_test")[:300]
    with np.load(REPO/"data/train.npz") as three, np.load(RUN/"data/train.npz") as one:
        ids = three["index"][selected]
        np.testing.assert_array_equal(ids,one["index"][selected])
        iotas = np.column_stack([one["iotas"][selected,0],three["iotas"][selected]])
        fields = np.concatenate([one["y"][selected],three["y"][selected]],axis=1)
        valid = np.column_stack([one["K"][selected],three["K"][selected]])
    ranks = np.load(RUN/"data/current_rank.npy")[selected]
    assert fields.shape == (300,4,13,20)
    encoded = {}
    for mid,current,prefix,field,rank in zip(ids,iotas,valid,fields,ranks):
        np.testing.assert_array_equal(field[0,:prefix[1]],field[int(rank)+1,:prefix[1]])
        np.testing.assert_equal(current[0],current[int(rank)+1])
        shown=np.isfinite(field)
        assert np.count_nonzero(shown)==20*(prefix[0]+3*prefix[1])
        assert np.all((field[shown]>=0)&(field[shown]<=2))
        raw=np.where(shown,np.rint(np.nan_to_num(field)*10000),65535).astype("<u2")
        encoded[str(mid)]=[current.tolist(),prefix.tolist(),base64.b64encode(raw.tobytes()).decode("ascii")]
    DATA.write_text(json.dumps(encoded,separators=(",",":")),encoding="utf-8")


def fragment():
    template=Path(__file__).with_name("field_template.html").read_text(encoding="utf-8")
    return template.replace("__MEASURED_FIELDS__",DATA.read_text(encoding="utf-8"))


if __name__=="__main__":
    prepare()
