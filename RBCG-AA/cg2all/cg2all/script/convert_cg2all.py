#!/usr/bin/env python

import os
import sys
import json
import time
import tqdm
import pathlib
import argparse

import numpy as np

import torch
import dgl

os.environ["OPENMM_PLUGIN_DIR"] = "/dev/null"
import mdtraj

from cg2all.lib.libconfig import MODEL_HOME
from cg2all.lib.libdata import (
    PredictionData,
    create_trajectory_from_batch,
    create_topology_from_data,
    standardize_atom_name,
)
from cg2all.lib.residue_constants import read_coarse_grained_topology
import cg2all.lib.libcg
from cg2all.lib.libpdb import write_SSBOND
from cg2all.lib.libter import patch_termini
import cg2all.lib.libmodel

import warnings

warnings.filterwarnings("ignore")


def main():
    arg = argparse.ArgumentParser(prog="convert_cg2all")
    arg.add_argument("-p", "--pdb", dest="in_pdb_fn", required=True)
    arg.add_argument("-d", "--dcd", dest="in_dcd_fn", default=None)
    arg.add_argument("-o", "--out", "--output", dest="out_fn", required=True)
    arg.add_argument("-opdb", dest="outpdb_fn")
    arg.add_argument("-ref_traj", "-ref", dest="ref_traj", required=False)
    arg.add_argument(
        "--cg",
        dest="cg_model",
        default="CalphaBasedModel",
        # fmt:off
        choices=["CalphaBasedModel", "CA", "ca", \
                "ResidueBasedModel", "RES", "res", \
                "Martini", "martini", "Martini2", "martini2", \
                "Martini3", "martini3", \
                "PRIMO", "primo", \
                "BB", "bb", "backbone", "Backbone", "BackboneModel", \
                "MC", "mc", "mainchain", "Mainchain", "MainchainModel",
                "CACM", "cacm", "CalphaCM", "CalphaCMModel",
                "CASC", "casc", "CalphaSC", "CalphaSCModel",
                "SC", "sc", "sidechain", "SidechainModel",
                ]
        # fmt:on
    )
    arg.add_argument("--chain-break-cutoff", dest="chain_break_cutoff", default=10.0, type=float)
    arg.add_argument("-a", "--all", "--is_all", dest="is_all", default=False, action="store_true")
    arg.add_argument("--fix", "--fix_atom", dest="fix_atom", default=False, action="store_true")
    arg.add_argument("--standard-name", dest="standard_names", default=False, action="store_true")
    arg.add_argument("--ckpt", dest="ckpt_fn", default=None)
    arg.add_argument("--time", dest="time_json", default=None)
    arg.add_argument("--device", dest="device", default=None)
    arg.add_argument("--batch", dest="batch_size", default=1, type=int)
    arg.add_argument(
        "--proc", dest="n_proc", default=int(os.getenv("OMP_NUM_THREADS", 1)), type=int
    )
    arg = arg.parse_args()
    #
    timing = {}
    #
    timing["loading_ckpt"] = time.time()
    if arg.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(arg.device)

    if arg.ckpt_fn is None:
        if arg.cg_model is None:
            raise ValueError("Either --cg or --ckpt argument should be given.")
        else:
            # fmt:off
            if arg.cg_model in ["CalphaBasedModel", "CA", "ca"]:
                model_type = "CalphaBasedModel"
            elif arg.cg_model in ["ResidueBasedModel", "RES", "res"]:
                model_type = "ResidueBasedModel"
            elif arg.cg_model in ["Martini", "martini", "Martini2", "martini2"]:
                model_type = "Martini"
            elif arg.cg_model in ["Martini3", "martini3"]:
                model_type = "Martini3"
            elif arg.cg_model in ["PRIMO", "primo"]:
                model_type = "PRIMO"
            elif arg.cg_model in ["CACM", "cacm", "CalphaCM", "CalphaCMModel"]:
                model_type = "CalphaCMModel"
            elif arg.cg_model in ["CASC", "casc", "CalphaSC", "CalphaSCModel"]:
                model_type = "CalphaSCModel"
            elif arg.cg_model in ["SC", "sc", "sidechain", "SidechainModel"]:
                model_type = "SidechainModel"
            elif arg.cg_model in ["BB", "bb", "backbone", "Backbone", "BackboneModel"]:
                model_type = "BackboneModel"
            elif arg.cg_model in ["MC", "mc", "mainchain", "Mainchain", "MainchainModel"]:
                model_type = "MainchainModel"
            # fmt:on
            #
            if arg.fix_atom:
                arg.ckpt_fn = MODEL_HOME / f"{model_type}-FIX.ckpt"
            else:
                arg.ckpt_fn = MODEL_HOME / f"{model_type}.ckpt"
        #
        if not arg.ckpt_fn.exists():
            cg2all.lib.libmodel.download_ckpt_file(model_type, arg.ckpt_fn, fix_atom=arg.fix_atom)
    #
    ckpt = torch.load(arg.ckpt_fn, map_location=device)
    config = ckpt["hyper_parameters"]
    timing["loading_ckpt"] = time.time() - timing["loading_ckpt"]
    #
    timing["model_configuration"] = time.time()
    if config["cg_model"] == "CalphaBasedModel":
        cg_model = cg2all.lib.libcg.CalphaBasedModel
    elif config["cg_model"] == "ResidueBasedModel":
        cg_model = cg2all.lib.libcg.ResidueBasedModel
    elif config["cg_model"] == "Martini":
        cg_model = cg2all.lib.libcg.Martini
    elif config["cg_model"] == "Martini3":
        cg_model = cg2all.lib.libcg.Martini3
    elif config["cg_model"] == "PRIMO":
        cg_model = cg2all.lib.libcg.PRIMO
    elif config["cg_model"] == "CalphaCMModel":
        cg_model = cg2all.lib.libcg.CalphaCMModel
    elif config["cg_model"] == "CalphaSCModel":
        cg_model = cg2all.lib.libcg.CalphaSCModel
    elif config["cg_model"] == "SidechainModel":
        cg_model = cg2all.lib.libcg.SidechainModel
    elif config["cg_model"] == "BackboneModel":
        cg_model = cg2all.lib.libcg.BackboneModel
    elif config["cg_model"] == "MainchainModel":
        cg_model = cg2all.lib.libcg.MainchainModel
    #
    if arg.is_all and config["cg_model"] in ["PRIMO", "Martini", "Martini3"]:
        topology_map = read_coarse_grained_topology(config["cg_model"].lower())
    else:
        topology_map = None
    #
    config = cg2all.lib.libmodel.set_model_config(config, cg_model, flattened=False)
    model = cg2all.lib.libmodel.Model(config, cg_model, compute_loss=False)
    #
    state_dict = ckpt["state_dict"]
    for key in list(state_dict):
        state_dict[".".join(key.split(".")[1:])] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.set_constant_tensors(device)
    model.eval()
    timing["model_configuration"] = time.time() - timing["model_configuration"]
    #
    timing["loading_input"] = time.time()
    input_s = PredictionData(
        arg.in_pdb_fn,
        cg_model,
        topology_map=topology_map,
        dcd_fn=arg.in_dcd_fn,
        radius=config.globals.radius,
        chain_break_cutoff=0.1 * arg.chain_break_cutoff,
        is_all=arg.is_all,
        fix_atom=True,#config.globals.fix_atom,
        batch_size=arg.batch_size,
    )
    #
    if arg.in_dcd_fn is not None:
        n_frame0 = input_s.n_frame0
        unitcell_lengths = input_s.cg.unitcell_lengths
        unitcell_angles = input_s.cg.unitcell_angles
    #
    if len(input_s) > 1 and (arg.n_proc > 1 or arg.batch_size > 1):
        input_s = dgl.dataloading.GraphDataLoader(
            input_s, batch_size=arg.batch_size, num_workers=arg.n_proc, shuffle=False
        )
    else:
        input_s = dgl.dataloading.GraphDataLoader(
            input_s, batch_size=1, num_workers=1, shuffle=False
        )
    timing["loading_input"] = time.time() - timing["loading_input"]
    #
    if arg.in_dcd_fn is None:
        t0 = time.time()
        batch = next(iter(input_s)).to(device)
        timing["loading_input"] += time.time() - t0
        #
        t0 = time.time()
        with torch.no_grad():
            R = model.forward(batch)[0]["R"]
        timing["forward_pass"] = time.time() - t0
        #
        timing["writing_output"] = time.time()
        traj_s, ssbond_s = create_trajectory_from_batch(batch, R)
        output = patch_termini(traj_s[0])
        if arg.standard_names:
            standardize_atom_name(output)
        output.save(arg.out_fn)
        if len(ssbond_s[0]) > 0:
            write_SSBOND(arg.out_fn, output.top, ssbond_s[0])
        timing["writing_output"] = time.time() - timing["writing_output"]
    else:
        timing["forward_pass"] = 0.0
        xyz = []
        t0 = time.time()
        for batch in tqdm.tqdm(input_s, total=len(input_s)):
            batch = batch.to(device)
            timing["loading_input"] += time.time() - t0
            #
            t0 = time.time()
            with torch.no_grad():
                R = model.forward(batch)[0]["R"].cpu().detach().numpy()
                mask = batch.ndata["output_atom_mask"].cpu().detach().numpy()
                xyz.append(R[mask > 0.0])
            timing["forward_pass"] += time.time() - t0
            t0 = time.time()
        #
        timing["writing_output"] = time.time()
        if arg.batch_size > 1:
            batch = dgl.unbatch(batch)[0]
            xyz = np.concatenate(xyz, axis=0)
            xyz = xyz.reshape((n_frame0, -1, 3))
        else:
            xyz = np.array(xyz)
        top, atom_index = create_topology_from_data(batch)
        xyz = xyz[:, atom_index]
        # compare to reference trajectory
        # ref_xyz = mdtraj.load(arg.ref_traj).xyz
        # ref_xyz = ref_xyz[:, atom_index]
        # dxyz = (xyz - ref_xyz)
        # rmsd_all = np.sqrt(np.power(dxyz, 2).sum(-1).mean())
        traj = mdtraj.Trajectory(
            xyz=xyz,
            topology=top,
            unitcell_lengths=unitcell_lengths,
            unitcell_angles=unitcell_angles,
        )
        output = patch_termini(traj)
        if arg.standard_names:
            standardize_atom_name(output)
        output.save(arg.out_fn)
        #
        if arg.outpdb_fn is not None:
            output[-1].save(arg.outpdb_fn)
        #
        timing["writing_output"] = time.time() - timing["writing_output"]

    time_total = 0.0
    for step, t in timing.items():
        time_total += t
    timing["total"] = time_total
    #
    print(timing)
    if arg.time_json is not None:
        with open(arg.time_json, "wt") as fout:
            fout.write(json.dumps(timing, indent=2))
    print("RMSD:", rmsd_all)

if __name__ == "__main__":
    main()
