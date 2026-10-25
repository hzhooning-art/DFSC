"""Frozen practicality gate: observation dead time and positive input envelopes."""
from pathlib import Path
import json
import time

import numpy as np

from delayed_rectangle_control import rectangle_kernel
from observation_weighted_prediction import anchor, digest, integrate_truth, solve

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "experiments/specs/dead_time_input_precision_20261012.json"
OUT = ROOT / "results/p4_dead_time_input_precision_20261012.json"
ARCHIVE = OUT.with_suffix(".npz")


def base_kernel(rate, design):
    if design == "constant":
        return anchor(rate)
    if design == "delayed_rectangle":
        return rectangle_kernel(rate)
    raise ValueError(design)


def target_kernel(rate, wait):
    rate = np.asarray(rate, dtype=float)
    safe = np.where(rate == 0, 1., rate)
    return np.where(rate == 0, 1., -2*np.expm1(-rate/2)/safe)*np.exp(-wait*rate)


def target_ratio(rate, design, excess_wait):
    if excess_wait < 0:
        raise ValueError("Unbounded target cannot use a bounded-tail representation")
    rate = np.asarray(rate, dtype=float)
    if design == "constant":
        basic = 2/(1+np.exp(-rate/2))
    elif design == "delayed_rectangle":
        basic = target_kernel(rate, 0)/rectangle_kernel(rate)
    else:
        raise ValueError(design)
    return basic*np.exp(-excess_wait*rate)


def representation(spec, design, dead_time, wait, cells):
    excess = wait-dead_time
    if excess < 0:
        raise ValueError("Prediction is unbounded")
    edges = np.r_[0., np.geomspace(spec["minimum_positive_rate"], spec["tail_start"], cells-1), np.inf]
    nodes = edges[:-1]
    delays = np.array(spec["observation_delays_from_first"])
    obshi = np.exp(-delays[:, None]*edges[:-1])
    obslo = np.empty_like(obshi)
    obslo[:, :-1] = np.exp(-delays[:, None]*edges[1:-1])
    obslo[:, -1] = (delays == 0).astype(float)
    qleft = target_ratio(edges[:-1], design, excess)
    qright = np.empty(cells)
    qright[:-1] = target_ratio(edges[1:-1], design, excess)
    qright[-1] = (2. if design == "constant" else 2/3) if excess == 0 else 0.
    qlo, qhi = np.minimum(qleft, qright), np.maximum(qleft, qright)
    critical = None
    # log derivative of 2 exp(-e lambda)/(1+exp(-lambda/2)) is
    # 1/(2(exp(lambda/2)+1))-e, strictly decreasing.
    if design == "constant" and 0 < excess < .25:
        critical = 2*np.log(.5/excess-1)
        crossed = (edges[:-1] <= critical) & (critical <= edges[1:])
        qhi[crossed] = float(target_ratio(critical, design, excess))
    qlo = np.maximum(0., qlo-2e-12)
    qhi += 2e-12
    exact = np.exp(-delays[:, None]*nodes)
    return {"edges":edges, "nodes":nodes, "outer_a":np.vstack((obslo,-obshi)),
            "inner_a":np.vstack((exact,-exact)), "q_low":qlo, "q_high":qhi,
            "inner_q":target_ratio(nodes,design,excess), "critical_rate":critical}


def relative_radius(interval):
    low, high = interval
    if low <= 0:
        return 1.
    return (high-low)/(high+low)


def robust_interval(interval, rho):
    if not 0 <= rho < 1:
        raise ValueError("Relative envelope must be in [0,1)")
    return [interval[0]/(1+rho), interval[1]/(1-rho)]


def decisions(rows, spec):
    result = []
    for row in rows:
        if row["cells"] != max(spec["cell_counts"]):
            continue
        for rho in spec["input_relative_envelopes"]:
            inner = robust_interval(row["inner_interval"],rho)
            outer = robust_interval(row["outer_interval"],rho)
            widths = [inner[1]-inner[0],outer[1]-outer[0]]
            rel = [relative_radius(inner),relative_radius(outer)]
            def state(bracket, limit):
                return "accept" if bracket[1] <= limit-spec["decision_margin"] else "reject" if bracket[0] > limit+spec["decision_margin"] else "unresolved"
            abs_state=state(widths,spec["absolute_width_limit"])
            rel_state=state(rel,spec["relative_minimax_radius_limit"])
            result.append({"truth":row["truth"],"design":row["design"],"dead_time":row["dead_time"],"wait":row["wait"],
                           "rho":rho,"inner_interval":inner,"outer_interval":outer,"width_bracket":widths,
                           "relative_radius_bracket":rel,"absolute_state":abs_state,"relative_state":rel_state,
                           "both_pass":abs_state==rel_state=="accept"})
    summaries=[]
    for design in spec["designs"]:
        for dead in spec["reference_dead_times"]:
            for rho in spec["input_relative_envelopes"]:
                counts=[]
                for wait in spec["target_waits"]:
                    rr=[r for r in result if (r["design"],r["dead_time"],r["rho"],r["wait"])==(design,dead,rho,wait)]
                    counts.append({"wait":wait,"bounded":wait>=dead,
                                   "absolute_pass":sum(r["absolute_state"]=="accept" for r in rr),
                                   "relative_pass":sum(r["relative_state"]=="accept" for r in rr),
                                   "both_pass":sum(r["both_pass"] for r in rr),"case_count":len(rr)})
                passing=[c["wait"] for c in counts if c["both_pass"]==4]
                summaries.append({"design":design,"dead_time":dead,"rho":rho,"wait_counts":counts,
                                  "first_tested_all_case_wait":min(passing) if passing else None})
    gate=all(any(r["first_tested_all_case_wait"] is not None for r in summaries
                 if r["design"]=="delayed_rectangle" and r["dead_time"]==dead and r["rho"]==.03)
             for dead in spec["reference_dead_times"])
    return result,summaries,gate


def main():
    if OUT.exists() or ARCHIVE.exists():
        raise FileExistsError("Frozen outputs already exist")
    start=time.perf_counter()
    spec=json.loads(SPEC.read_text(encoding="utf-8"))
    parent=ROOT/"results"/spec["parent_result"]
    assert digest(parent)==spec["parent_result_sha256"]
    paper=ROOT/"paper/positive_hereditary_prediction.tex"
    assert digest(paper)==spec["parent_manuscript_sha256"]
    truthpath=SPEC.parent/spec["parent_truth_spec"]
    truths=json.loads(truthpath.read_text(encoding="utf-8"))["truths"]
    rows,data,witnesses,logs,payload=[],[],[],[],{}
    reps={(d,dt,w,c):representation(spec,d,dt,w,c) for d in spec["designs"]
          for dt in spec["reference_dead_times"] for w in spec["target_waits"] if w>=dt for c in spec["cell_counts"]}
    for truth in truths:
        for design in spec["designs"]:
            for dead in spec["reference_dead_times"]:
                obs=np.array([integrate_truth(truth,lambda rate:float(base_kernel(rate,design))*np.exp(-(dead+d)*rate))
                              for d in spec["observation_delays_from_first"]])
                data.append({"truth":truth["name"],"design":design,"dead_time":dead,"observations":obs.tolist()})
                b=spec["absolute_response_band"]
                rhs=np.r_[obs+b,-obs+b]
                for wait in spec["target_waits"]:
                    actual=integrate_truth(truth,lambda rate:float(target_kernel(rate,wait)))
                    if wait<dead:
                        for rate in spec["adversarial_rates"]:
                            g=float(base_kernel(rate,design))*np.exp(-dead*rate)
                            mass=spec["adversarial_anchor_increment"]/g
                            changes=spec["adversarial_anchor_increment"]*np.exp(-rate*np.array(spec["observation_delays_from_first"]))
                            change=mass*float(target_kernel(rate,wait))
                            assert np.all(np.isfinite([mass,change])) and max(changes)<b
                            witnesses.append({"truth":truth["name"],"design":design,"dead_time":dead,"wait":wait,
                                              "rate":rate,"physical_added_mass":mass,"observation_changes":changes.tolist(),
                                              "target_change":change,"truth_target":actual})
                        continue
                    for cells in spec["cell_counts"]:
                        rep=reps[design,dead,wait,cells]
                        stem=f"{truth['name']}_{design}_d{dead}_w{wait}_{cells}"
                        ol=solve(rep["q_low"],rep["outer_a"],rhs,stem+"_outer_low",payload,logs,spec["lp_replay_tolerance"])
                        oh=-solve(-rep["q_high"],rep["outer_a"],rhs,stem+"_outer_high",payload,logs,spec["lp_replay_tolerance"])
                        il=solve(rep["inner_q"],rep["inner_a"],rhs,stem+"_inner_low",payload,logs,spec["lp_replay_tolerance"])
                        ih=-solve(-rep["inner_q"],rep["inner_a"],rhs,stem+"_inner_high",payload,logs,spec["lp_replay_tolerance"])
                        assert ol-2e-7<=il<=ih<=oh+2e-7 and ol-2e-7<=actual<=oh+2e-7
                        rows.append({"truth":truth["name"],"design":design,"dead_time":dead,"wait":wait,"cells":cells,
                                     "outer_interval":[ol,oh],"inner_interval":[il,ih],"truth_target":actual,
                                     "endpoint_bracket":max(il-ol,oh-ih),"label":stem})
                print(json.dumps({"truth":truth["name"],"design":design,"dead_time":dead,"LPs":len(logs)}),flush=True)
    selected,summaries,gate=decisions(rows,spec)
    fine=[r for r in rows if r["cells"]==max(spec["cell_counts"])]
    precision=max(r["endpoint_bracket"] for r in fine)<=spec["endpoint_bracket_tolerance"]
    np.savez_compressed(ARCHIVE,**payload)
    result={"schema":"P4-dead-time-input-precision-result-v1","spec_sha256":digest(SPEC),"source_sha256":digest(__file__),
            "parent_result_sha256":digest(parent),"parent_manuscript_sha256":spec["parent_manuscript_sha256"],
            "truth_spec_sha256":digest(truthpath),"archive_sha256":digest(ARCHIVE),
            "dependencies_sha256":{n:digest(ROOT/"experiments"/n) for n in ("delayed_rectangle_control.py","observation_weighted_prediction.py")},
            "data":data,"rows":rows,"unbounded_witnesses":witnesses,"lp_logs":logs,"decisions":selected,"summaries":summaries,
            "all_internal_checks_pass":True,"precision_gate_pass":precision,"continuation_gate_pass":gate,
            "max_finest_endpoint_bracket":max(r["endpoint_bracket"] for r in fine),
            "duration_seconds":time.perf_counter()-start,"limits":spec["limits"]}
    OUT.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"LPs":len(logs),"nominal_rows":len(rows),"robust_decisions":len(selected),
                      "unbounded_pairs":len(witnesses),"precision_gate":precision,"continuation_gate":gate,
                      "summaries":[{k:v for k,v in r.items() if k!="wait_counts"} for r in summaries],
                      "duration_seconds":result["duration_seconds"]}),flush=True)


if __name__=="__main__":
    main()
