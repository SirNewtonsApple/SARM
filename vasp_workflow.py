# vasp_workflow.py
from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class VaspWorkflow:
    """
    Simple JSON-driven VASP workflow.

    Core idea (simple, deterministic):
      For EVERY run:
        1) (Re)construct input files first (INCAR, POSCAR, KPOINTS, POTCAR, and any extra files)
        2) Apply user-defined copy rules (source -> destination) to overwrite/bring restart files
           (e.g., copy CONTCAR->POSCAR, copy WAVECAR/CHGCAR, etc.)
        3) Run VASP using user command
        4) Backup the calculation directory

    This avoids trying to infer "fresh vs continuation" in code; the JSON copy rules define it.
    """

    # ----------------------------
    # Construction / config
    # ----------------------------
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.calc_dir = Path(self.cfg["calc_dir"])

        # cached per-prepare (computed from JSON)
        self._species_order: Optional[List[str]] = None

    @classmethod
    def from_json(cls, json_path: str | Path) -> "VaspWorkflow":
        cfg = json.loads(Path(json_path).read_text(encoding="utf-8"))
        return cls(cfg)

    # ----------------------------
    # Public: one full run cycle
    # ----------------------------
    def run(
        self,
        tag: str,
        incar_updates: Optional[Dict[str, Any]] = None,
        extra_copy_rules: Optional[List[Dict[str, Any]]] = None,
        command_override: Optional[str] = None,
        command_mode_override: Optional[str] = None,
    ) -> Path:
        """
        One complete cycle:
          prepare_inputs() -> apply_copy_rules() -> run_vasp() -> backup()

        Returns: backup directory path.
        """
        self.prepare_inputs(incar_updates=incar_updates)
        self.apply_copy_rules(extra_copy_rules=extra_copy_rules)
        self.run_vasp(command_override=command_override, command_mode_override=command_mode_override)
        return self.backup(tag)

    # ----------------------------
    # Step 1: construct inputs (always)
    # ----------------------------
    def prepare_inputs(self, incar_updates: Optional[Dict[str, Any]] = None) -> None:
        """
        Always (re)construct these before a run:
          - POSCAR  (from random FCC alloy unless you later overwrite it via copy rules)
          - INCAR   (from cfg + optional updates)
          - files in cfg["vasp"]["files"]  (e.g., KPOINTS, ICONST, INWAV, ...)
          - POTCAR  (concatenate element POTCARs based on species order)
        """
        self.calc_dir.mkdir(parents=True, exist_ok=True)

        # POSCAR (generated)
        self._write_poscar_random_fcc()

        # INCAR (generated)
        incar = dict(self.cfg["vasp"]["incar"])
        if incar_updates:
            incar.update(incar_updates)
        self._write_incar(self.calc_dir / "INCAR", incar)

        # Other files like KPOINTS/ICONST (generated or copied from a path)
        self._write_or_copy_vasp_files()

        # POTCAR (generated)
        self._build_potcar()

    def _write_poscar_random_fcc(self) -> None:
        """
        Build a random FCC alloy POSCAR from:
          cfg["formula"]["counts"]  : {"Fe":82,"Si":4,...}
          cfg["formula"]["host"]    : "Fe"
          cfg["formula"]["species_order"] (optional)
          cfg["structure"]["lattice_constant"], cfg["structure"]["seed"]
        """
        fb = self.cfg["formula"]
        counts: Dict[str, int] = {str(k): int(v) for k, v in fb["counts"].items()}
        host: str = str(fb.get("host", "Fe"))
        if host not in counts:
            raise ValueError(f"Host '{host}' must appear in formula.counts.")

        explicit_order = fb.get("species_order")
        species_order = self._species_order_from_counts(counts, host=host, explicit_order=explicit_order)
        self._species_order = species_order

        st = self.cfg["structure"]
        if st.get("type", "fcc_random_alloy") != "fcc_random_alloy":
            raise ValueError("Only structure.type='fcc_random_alloy' is implemented.")
        a = float(st["lattice_constant"])
        seed = int(st.get("seed", 0))

        lattice, symbols, frac_coords = self._build_random_fcc_alloy(counts, a, host, seed)

        comment = self.cfg.get("poscar_comment", "random FCC")
        self._write_poscar_file(
            path=self.calc_dir / "POSCAR",
            comment=comment,
            lattice=lattice,
            species_order=species_order,
            symbols=symbols,
            frac_coords=frac_coords,
        )

    def _species_order_from_counts(
        self,
        counts: Dict[str, int],
        host: Optional[str],
        explicit_order: Optional[List[str]],
    ) -> List[str]:
        elements = set(counts.keys())

        if explicit_order:
            missing = [e for e in explicit_order if e not in elements]
            extra = [e for e in elements if e not in explicit_order]
            if missing:
                raise ValueError(f"species_order contains unknown elements: {missing}")
            if extra:
                raise ValueError(f"species_order missing elements: {extra}")
            return list(explicit_order)

        if host and host in elements:
            rest = sorted([e for e in elements if e != host])
            return [host] + rest
        return sorted(elements)

    def _build_random_fcc_alloy(
        self,
        counts: Dict[str, int],
        lattice_constant: float,
        host: str,
        seed: int,
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Returns:
          lattice_vectors (3,3) in Angstrom
          symbols list[str] length N
          fractional coords (N,3)
        """
        total = int(sum(counts.values()))
        a = float(lattice_constant)
        rng = np.random.default_rng(int(seed))

        # FCC conventional cell has 4 sites
        n = max(1, int(math.ceil((total / 4.0) ** (1.0 / 3.0))))
        parent_sites = 4 * n**3

        basis = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            dtype=float,
        )

        coords = np.empty((parent_sites, 3), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    shift = np.array([i, j, k], dtype=float)
                    for b in basis:
                        coords[idx] = (shift + b) / n
                        idx += 1

        # Trim to exact number of atoms by removing random sites
        if parent_sites > total:
            keep = np.ones(parent_sites, dtype=bool)
            remove_idx = rng.choice(parent_sites, size=(parent_sites - total), replace=False)
            keep[remove_idx] = False
            coords = coords[keep]

        # Assign species: host everywhere then substitute others
        symbols = [host] * len(coords)
        site_ids = np.arange(len(coords))
        rng.shuffle(site_ids)

        cursor = 0
        for el, num in counts.items():
            if el == host:
                continue
            chosen = site_ids[cursor : cursor + num]
            if len(chosen) != num:
                raise RuntimeError(f"Not enough host sites to place {el} x{num}")
            for s in chosen:
                symbols[int(s)] = el
            cursor += num

        # Cubic supercell lattice vectors
        L = np.array([[a * n, 0, 0], [0, a * n, 0], [0, 0, a * n]], dtype=float)
        return L, symbols, coords

    def _write_poscar_file(
        self,
        path: Path,
        comment: str,
        lattice: np.ndarray,
        species_order: List[str],
        symbols: List[str],
        frac_coords: np.ndarray,
    ) -> None:
        sym_arr = np.array(symbols, dtype=object)

        coords_by_el: List[np.ndarray] = []
        counts: List[int] = []
        for el in species_order:
            sel = (sym_arr == el)
            c = frac_coords[sel]
            coords_by_el.append(c)
            counts.append(int(c.shape[0]))

        with path.open("w", encoding="utf-8") as f:
            f.write(f"{comment}\n")
            f.write("1.0\n")
            for v in lattice:
                f.write(f"  {v[0]:16.10f} {v[1]:16.10f} {v[2]:16.10f}\n")
            f.write("  " + "  ".join(species_order) + "\n")
            f.write("  " + "  ".join(str(c) for c in counts) + "\n")
            f.write("Direct\n")
            for c in coords_by_el:
                for x, y, z in c:
                    f.write(f"  {x:16.10f} {y:16.10f} {z:16.10f}\n")

    def _write_incar(self, path: Path, incar: Dict[str, Any]) -> None:
        keys = sorted(incar.keys(), key=lambda k: k.upper())
        with path.open("w", encoding="utf-8") as f:
            for k in keys:
                v = incar[k]
                if v is None:
                    continue
                f.write(f"{k.upper():<16} = {self._format_incar_value(v)}\n")

    def _format_incar_value(self, v: Any) -> str:
        if isinstance(v, bool):
            return ".TRUE." if v else ".FALSE."
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, (list, tuple)):
            return " ".join(str(x) for x in v)
        return str(v)  # strings like "4*5.0" are valid

    def _write_or_copy_vasp_files(self) -> None:
        """
        vasp.files: {
          "KPOINTS": {"content": "..."} or {"path": "..."},
          "ICONST":  {"path": "..."},
          ...
        }
        These are always written/copied fresh before each run.
        """
        files = self.cfg["vasp"].get("files", {})
        for filename, spec in files.items():
            dst = self.calc_dir / filename
            if "content" in spec:
                dst.write_text(spec["content"], encoding="utf-8")
            elif "path" in spec:
                shutil.copy2(Path(spec["path"]), dst)
            else:
                raise ValueError(f"File spec must include 'content' or 'path': {filename} -> {spec}")

    def _build_potcar(self) -> None:
        """
        Build POTCAR by concatenating element POTCARs in species order.
        Always done fresh before each run.
        """
        if not self._species_order:
            raise RuntimeError("Species order not set; prepare_inputs() must be called first.")

        pot = self.cfg["vasp"]["potcar"]
        root = Path(pot["root"])
        mapping: Dict[str, str] = pot.get("map", {})  # e.g. {"Fe": "Fe_pv"}
        template: str = pot.get("template", "{root}/{label}/POTCAR")

        out = self.calc_dir / "POTCAR"
        parts: List[bytes] = []
        for el in self._species_order:
            label = mapping.get(el, el)
            p = Path(template.format(root=str(root), label=label))
            if not p.is_file():
                raise FileNotFoundError(f"Missing POTCAR for element {el} (label={label}): {p}")
            parts.append(p.read_bytes())

        out.write_bytes(b"".join(parts))

    # ----------------------------
    # Step 2: apply copy rules (after construction, before run)
    # ----------------------------
    def apply_copy_rules(self, extra_copy_rules: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Apply copy rules AFTER constructing inputs.

        Rules come from:
          - cfg["copy_rules"] (list)
          - plus extra_copy_rules passed at runtime

        Rule format (examples):
          {"src": "/old/run/CONTCAR", "dst": "POSCAR", "optional": false}
          {"src": "/old/run/WAVECAR", "dst": "WAVECAR", "optional": true}
          {"src_dir": "/old/run", "src_name": "CHGCAR", "dst": "CHGCAR", "optional": true}

        This is the unified mechanism for:
          - continuation runs (copy CONTCAR->POSCAR, copy WAVECAR/CHGCAR)
          - copying any extra prepared files
          - overwriting KPOINTS/ICONST if you choose
        """
        rules: List[Dict[str, Any]] = []
        rules += list(self.cfg.get("copy_rules", []))
        if extra_copy_rules:
            rules += list(extra_copy_rules)

        for r in rules:
            optional = bool(r.get("optional", False))

            if "src" in r:
                src = Path(r["src"])
            else:
                src_dir = Path(r["src_dir"])
                src_name = r["src_name"]
                src = src_dir / src_name

            dst = self.calc_dir / r["dst"]

            if not (src.is_file() and src.stat().st_size > 0):
                if optional:
                    continue
                raise FileNotFoundError(f"Required copy source missing/empty: {src}")

            shutil.copy2(src, dst)

    # ----------------------------
    # Step 3: run VASP with user command
    # ----------------------------
    def run_vasp(self, command_override: Optional[str] = None, command_mode_override: Optional[str] = None) -> None:
        """
        Run user-specified command in calc_dir.

        Uses:
          cfg["vasp"]["command"], cfg["vasp"]["command_mode"] unless overridden.
        """
        vcfg = self.cfg["vasp"]
        cmd = os.path.expandvars(command_override or vcfg["command"])
        mode = (command_mode_override or vcfg.get("command_mode", "exec")).lower()

        if mode == "bash":
            #subprocess.run(["bash", "-lc", cmd], cwd=str(self.calc_dir), check=True)
            subprocess.run(cmd, cwd=str(self.calc_dir), shell=True, text=True, check=True)
            return
        if mode == "exec":
            args = shlex.split(cmd)
            subprocess.run(args, cwd=str(self.calc_dir), check=True)
            return

        raise ValueError(f"Unknown command_mode: {mode} (use 'exec' or 'bash')")

    # ----------------------------
    # Step 4: backup
    # ----------------------------
    def backup(self, tag: str) -> Path:
        """
        backup: {"enabled": true, "root": "backups", "exclude": ["WAVECAR"]}
        """
        b = self.cfg.get("backup", {"enabled": True, "root": "backups", "exclude": []})
        if not b.get("enabled", True):
            return self.calc_dir

        root = Path(b.get("root", "backups"))
        root.mkdir(parents=True, exist_ok=True)
        dst = root / f"{self.calc_dir.name}.{tag}"

        exclude = set(b.get("exclude", []))

        def ignore_fn(_src: str, names: List[str]):
            return {n for n in names if n in exclude}

        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(self.calc_dir, dst, ignore=ignore_fn if exclude else None)
        return dst
