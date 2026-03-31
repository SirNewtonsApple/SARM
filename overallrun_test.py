import json
from vasp_workflow import VaspWorkflow

vasp_cfg = "project_test.json"

def main():
    wf = VaspWorkflow.from_json(vasp_cfg)

    wf.run(
        tag="T3000",
        incar_updates={"TEBEG": 3000},
        extra_copy_rules=[]  # nothing copied; inputs constructed fresh
    )

    for iMD in range(17+1):
        T = 2000 - 100 * iMD
        prev = wf.calc_dir  # same directory; you could also point to a backup folder
        wf.run(
            tag="T"+str(T),
            incar_updates={"TEBEG": T, "ICHARG": 1},
            extra_copy_rules=[
                {"src": str(prev / "CONTCAR"), "dst": "POSCAR", "optional": False},
            ],
        )

    print("Done.")

if __name__ == "__main__":
    main()
