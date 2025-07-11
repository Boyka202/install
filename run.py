import numpy as np
import subprocess
import damask
import shutil
import os
import json
import yaml
import logging
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
import sys

# ==================================
#          LOGGING SETUP
# ==================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization_damask3_TM.log'),
        logging.StreamHandler()
    ]
)

# ==================================
#          CONFIGURATION
# ==================================
class Config:
    """
    Configuration for Isothermal High-Temperature Calibration
    using DAMASK_grid and a 'dislotwin' model.
    """
    # =====================
    # SIMULATION SETTINGS
    # =====================
    MAX_PARALLEL_JOBS = 1
    CORES_PER_JOB = 100 # Adjust based on your system
    MAX_ITER = 10       # Number of optimization iterations

    # ===========================
    #  OBJECTIVE FUNCTION WEIGHTS
    # ===========================
    STRESS_ERROR_WEIGHT = 0.5
    HARDENING_ERROR_WEIGHT = 0.5

    # =============================
    #    FILES / DIRECTORIES
    # =============================
    BASE_DIR = Path(os.getcwd())
    RESULTS_DIR = BASE_DIR / "optimization_results_damask3_TM"
    PLOTS_DIR = RESULTS_DIR / "plots"
    MATERIAL_TEMPLATE = BASE_DIR / "material.yaml"
    GEOM_FILE = BASE_DIR / "geom.vti"
    # Using the loadcase you provided and confirmed works
    LOAD_FILE = BASE_DIR / "tensionX.yaml"
    EXP_DATA_FILE = BASE_DIR / "exp.txt"

    # =============================
    #   PARAMETER SPACE (dislotwin model)
    # =============================
    # These parameters from your 'dislotwin' model will be optimized.
    # !!! REVIEW AND ADJUST THESE BOUNDS BASED ON YOUR MATERIAL KNOWLEDGE !!!
    PARAM_SPACE = [
        Real(name='tau_0', low=23e6, high=26e6, prior='log-uniform', name='Peierls Stress (tau_0)'),
        Real(name='Q_sl', low=1.0e-19, high=2.0e-19 , prior='log-uniform', name='Activation Energy (Q_sl)'),
        Real(name='p_sl', low=0.2, high=0.4, name='Flow Rule Exponent (p_sl)'),
        Real(name='Q_cl', low=4.0e-19, high=5.0e-19, name='enhance climb'),
        Real(name='q_sl', low=1.0, high=2.0, name='Flow Rule Exponent (q_sl)'),
        Real(name='D_a',  low=8.0, high=12.0, name='Annihilation Coeff (D_a)'),
        Real(name='rho_dip_0', low=2.0e11, high=5.0e11, prior='log-uniform', name='Initial Dipole Density')
    ]

    # --- Path within material.yaml to the dictionary of plastic parameters ---
    YAML_PLASTIC_PATH = ['phase', 'ferrite', 'mechanical', 'plastic']

    @classmethod
    def setup(cls):
        cls.RESULTS_DIR.mkdir(exist_ok=True)
        cls.PLOTS_DIR.mkdir(exist_ok=True)
        history_file = cls.RESULTS_DIR / "optimization_history.json"
        if not history_file.exists():
            with open(history_file, 'w') as f: json.dump({"iterations": []}, f)
        logging.info(f"Results will be stored in: {cls.RESULTS_DIR}")

# ==================================
#       I/O AND PREPARATION
# ==================================
def read_experimental_data():
    logging.info(f"Reading experimental data from {Config.EXP_DATA_FILE}...")
    try:
        data = np.genfromtxt(Config.EXP_DATA_FILE, delimiter='\t', skip_header=1)
        strain_percent = data[:, 1] * 100 # Converts strain from 0-1 to 0-100
        stress_mpa = data[:, 0]
        return {'stress_strain': np.column_stack((strain_percent, stress_mpa))}
    except Exception as e:
        logging.error(f"Error reading experimental data: {e}"); raise

def update_material_config(params, run_dir):
    material_file_path = run_dir / Config.MATERIAL_TEMPLATE.name
    # Copy the template to the run directory to avoid modifying the original
    shutil.copy(Config.MATERIAL_TEMPLATE, material_file_path)
    logging.info(f"Updating material config in {material_file_path}")
    try:
        with open(material_file_path, 'r') as f: material_data = yaml.safe_load(f)
        current_level = material_data
        for key in Config.YAML_PLASTIC_PATH: current_level = current_level[key]
        for pname, pvalue in params.items():
            if pname in current_level:
                formatted_value = float(f'{pvalue:.6e}')
                if isinstance(current_level[pname], list):
                    current_level[pname] = [formatted_value] * len(current_level[pname])
                else: current_level[pname] = formatted_value
            else: logging.warning(f"Param '{pname}' not in material.yaml. Skipping.")
        with open(material_file_path, 'w') as f:
            yaml.dump(material_data, f, default_flow_style=None, sort_keys=False)
    except Exception as e:
        logging.error(f"Error during material config update: {e}", exc_info=True); raise

# ==================================
#       SIMULATION EXECUTION
# ==================================
def run_damask_simulation(run_dir, material_file_path_in_run_dir):
    """
    Run DAMASK_grid simulation, which you've confirmed works for your T-M setup.
    """
    logging.info(f"Starting DAMASK_grid simulation in {run_dir}")
    geom_file_src = Config.GEOM_FILE
    load_file_src = Config.LOAD_FILE
    material_file_src = Config.MATERIAL_TEMPLATE

    geom_file_dest = run_dir / geom_file_src.name
    load_file_dest = run_dir / load_file_src.name
    shutil.copy(geom_file_src, geom_file_dest)
    shutil.copy(load_file_src, load_file_dest)

    cmd = [
        'DAMASK_grid',
        '--load', load_file_dest.name,
        '--geom', geom_file_dest.name,
        '--material', material_file_path_in_run_dir.name
    ]
    logging.info(f"Executing command: {' '.join(map(str, cmd))}")

    # Construct the correct HDF5 filename: <geom>_<load>_<material>.hdf5
    output_base_name = f"{geom_file_src.stem}_{load_file_src.stem}_{material_file_src.stem}"
    hdf5_file = run_dir / f"{output_base_name}.hdf5"
    logging.info(f"Expecting HDF5 output at: {hdf5_file}")

    try:
        env = os.environ.copy()
        env['DAMASK_NUM_THREADS'] = str(Config.CORES_PER_JOB)
        process = subprocess.Popen(cmd, cwd=run_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        stdout_full, stderr_full = process.communicate()
        return_code = process.returncode

        if return_code == 0 and hdf5_file.exists():
            logging.info(f"DAMASK_grid completed successfully.")
            return True, hdf5_file
        else:
            logging.error(f"DAMASK_grid failed. Return code: {return_code}.")
            if not hdf5_file.exists(): logging.error(f"Expected output file {hdf5_file} was NOT created.")
            return False, None
    except Exception as e:
        logging.error(f"An error occurred running DAMASK_grid: {e}", exc_info=True)
        return False, None

# ==================================
#     POST-PROCESSING & OBJECTIVE
# ==================================
def process_damask_results(hdf5_file_path):
    logging.info(f"Processing DAMASK results from: {hdf5_file_path}")
    try:
        res = damask.Result(str(hdf5_file_path))
        res.add_stress_Cauchy(); res.add_strain()
        res.add_equivalent_Mises('Cauchy'); res.add_equivalent_Mises('epsilon_V^0.0(F)')
        incs = res.get_increments()
        if not incs: logging.error("No increments found."); return None
        s_key = next((k for k in res.keys() if 'Cauchy' in k and '_vM' in k), None)
        e_key = next((k for k in res.keys() if 'epsilon_V^0.0(F)' in k and '_vM' in k), None)
        if not all([s_key, e_key]):
            logging.error(f"Mises keys not found. Available: {list(res.keys())}"); return None
        avg_S = np.array([np.average(res.get(s_key)[i]) for i in incs]) / 1e6
        avg_s = np.array([np.average(res.get(e_key)[i]) for i in incs]) * 100
        sim_data = np.column_stack((avg_S, avg_s))
        logging.info(f"Successfully extracted {len(sim_data)} simulation data points.")
        return sim_data
    except Exception as e:
        logging.error(f"Error processing HDF5 file {hdf5_file_path}: {e}", exc_info=True)
        return None

def run_single_evaluation(params, exp_data, run_id):
    logging.info(f"\n{'='*10} Starting Evaluation {run_id} {'='*10}")
    logging.info(f"Parameters: {', '.join(f'{k}={v:.4e}' for k,v in params.items())}")
    run_dir = Config.RESULTS_DIR / f"run_{run_id}"; run_dir.mkdir(exist_ok=True)
    t_err, s_err, h_err = 1e7, 1e6, 1e6
    try:
        update_material_config(params, run_dir)
        success, hdf5 = run_damask_simulation(run_dir, run_dir / Config.MATERIAL_TEMPLATE.name)
        if success and hdf5:
            sim_data = process_damask_results(hdf5)
            if sim_data is not None and sim_data.size > 0:
                s_err = calculate_stress_error(sim_data, exp_data)
                h_err, exp_mid, exp_h, sim_h = calculate_hardening_error(sim_data, exp_data)
                t_err = (Config.STRESS_ERROR_WEIGHT * s_err + Config.HARDENING_ERROR_WEIGHT * h_err)
                logging.info(f"Eval {run_id} done: StressRMSE={s_err:.4f}, HardRMSE={h_err:.4f} ==> TotalObjective={t_err:.4f}")
                plot_current_results(sim_data, exp_data, params, run_id, exp_mid, exp_h, sim_h)
            else: logging.error(f"Evaluation {run_id} failed during results processing.")
        else: logging.error(f"Evaluation {run_id} failed during simulation execution.")
    except Exception as e:
        logging.exception(f"A critical error occurred during evaluation {run_id}: {e}")
    finally:
        update_optimization_history(params, s_err, h_err, t_err, run_id)
    return t_err

# ==================================
#       ERROR CALCULATION & PLOTTING
# ==================================
def calculate_stress_error(sim, exp):
    exp_s, exp_S = exp['stress_strain'][:, 0], exp['stress_strain'][:, 1]
    if sim.size == 0 or sim[-1, 1] < exp_s[0] or sim[0, 1] > exp_s[-1]: return 1e6
    sim_S_i = np.interp(exp_s, sim[:, 1], sim[:, 0], left=np.nan, right=np.nan)
    v = ~np.isnan(sim_S_i); diff = sim_S_i[v] - exp_S[v]
    if diff.size == 0: return 1e6
    rmse = np.sqrt(np.mean(diff**2))
    logging.info(f"Overall Stress RMSE: {rmse:.3f}")
    return rmse

def calculate_hardening_curve(s, S, step):
    n = len(s)
    if n <= step: return np.array([]), np.array([])
    v = np.where(np.diff(s) > 1e-9)[0] + 1; v = np.insert(v, 0, 0)
    s, S = s[v], S[v]
    if len(s) <= step: return np.array([]), np.array([])
    ds, dE = S[step:] - S[:-step], s[step:] - s[:-step]
    vd = dE > 1e-9
    if not np.any(vd): return np.array([]), np.array([])
    h = np.full_like(dE, np.nan); h[vd] = ds[vd] / dE[vd]
    mid_s = 0.5 * (s[:-step] + s[step:])
    return mid_s[vd], h[vd]

def calculate_hardening_error(sim, exp):
    exp_s, exp_S = exp['stress_strain'][:, 0], exp['stress_strain'][:, 1]
    exp_mid, exp_h = calculate_hardening_curve(exp_s, exp_S, step=5)
    if exp_mid.size == 0: return 1e6, *[np.array([])]*3
    if sim.size == 0: return 1e6, exp_mid, exp_h, np.full_like(exp_mid, np.nan)
    sim_s, sim_S = sim[:, 1], sim[:, 0]
    sim_mid, sim_h = calculate_hardening_curve(sim_s, sim_S, step=1)
    if sim_mid.size == 0: return 1e6, exp_mid, exp_h, np.full_like(exp_mid, np.nan)
    im = (exp_mid >= sim_mid[0]) & (exp_mid <= sim_mid[-1])
    exp_mid_i, exp_h_i = exp_mid[im], exp_h[im]
    if exp_mid_i.size == 0: return 1e6, exp_mid, exp_h, np.full_like(exp_mid, np.nan)
    sim_h_i = np.interp(exp_mid_i, sim_mid, sim_h)
    diff = sim_h_i - exp_h_i
    error = np.sqrt(np.mean(diff**2))
    logging.info(f"Overall Hardening RMSE: {error:.3f}")
    sim_h_plot = np.full_like(exp_mid, np.nan); sim_h_plot[im] = sim_h_i
    return error, exp_mid, exp_h, sim_h_plot

def plot_current_results(sim, exp, p, run_id, exp_hm, exp_hs, sim_h):
    fig, ax1 = plt.subplots(figsize=(12, 7))
    lines, labels = [], []
    le, = ax1.plot(exp['stress_strain'][:, 0], exp['stress_strain'][:, 1], 'b-', lw=2, label='Exp. S-S')
    lines.append(le); labels.append(le.get_label())
    if sim.size > 0:
        ls, = ax1.plot(sim[:, 1], sim[:, 0], 'r--', lw=2, label=f'Sim (Run {run_id}) S-S')
        lines.append(ls); labels.append(ls.get_label())
    ax1.set_xlabel("Strain (%)"); ax1.set_ylabel("Stress (MPa)", color='b')
    ax1.tick_params(axis='y', labelcolor='b'); ax1.grid(True, linestyle=':')
    ax2 = ax1.twinx()
    if exp_hm.size > 0:
        leh, = ax2.plot(exp_hm, exp_hs, 'g-', lw=2, label='Exp. Hardening')
        lines.append(leh); labels.append(leh.get_label())
        lsh, = ax2.plot(exp_hm, sim_h, 'm--', lw=2, label=f'Sim (Run {run_id}) Hardening')
        lines.append(lsh); labels.append(lsh.get_label())
        ax2.set_ylabel("Hardening Rate", color='g'); ax2.tick_params(axis='y', labelcolor='g')
    ax1.legend(lines, labels, loc='best');
    p_txt = "Parameters:\n" + "\n".join(f"{ps.name}: {v:.4e}" for ps, v in zip(Config.PARAM_SPACE, p.values()))
    plt.figtext(0.99, 0.5, p_txt, va='center', ha='left', bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    plt.title(f"DAMASK 3 T-M Opt: Run {run_id}"); plt.subplots_adjust(right=0.80)
    out_file = Config.PLOTS_DIR / f"comparison_run_{run_id}.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig); logging.info(f"Saved comparison plot: {out_file}")

def update_optimization_history(p, s_err, h_err, t_err, run_id):
    hf = Config.RESULTS_DIR / "optimization_history.json"
    try:
        with open(hf, 'r') as f: hist = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): hist = {"iterations": []}
    it_data = {"run_id": run_id, "timestamp": datetime.now().isoformat(),
               "parameters": {k: float(f'{v:.6e}') for k,v in p.items()},
               "stress_error_rmse": float(s_err),
               "hardening_error_rmse": float(h_err),
               "total_error_objective": float(t_err)}
    hist["iterations"].append(it_data)
    with open(hf, 'w') as f: json.dump(hist, f, indent=2)

def plot_optimization_progress():
    history_file = Config.RESULTS_DIR / "optimization_history.json"
    try:
        with open(history_file, 'r') as f: history_data = json.load(f)['iterations']
        if not history_data: return
        df = pd.DataFrame(history_data)
    except Exception as e:
        logging.error(f"Could not parse history file for plotting: {e}"); return
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(df['run_id'], df['stress_error_rmse'], 'b-o', label="Stress RMSE")
    axes[0].plot(df['run_id'], df['hardening_error_rmse'], 'r-s', label="Hardening RMSE")
    axes[0].plot(df['run_id'], df['total_error_objective'], 'k-x', ms=8, lw=2, label="Total Objective")
    axes[0].set_ylabel("Error / Objective"); axes[0].set_title("Optimization Progress")
    axes[0].legend(); axes[0].grid(True, linestyle=':');
    if any(e > 0 for e in df['total_error_objective']): axes[0].set_yscale('log')
    param_names = [p.name for p in Config.PARAM_SPACE]
    for name in param_names:
        axes[1].plot(df['run_id'], [p.get(name, np.nan) for p in df['parameters']], '-o', label=name)
    axes[1].set_xlabel("Iteration (Run ID)"); axes[1].set_ylabel("Parameter Value"); axes[1].set_title("Parameter Evolution")
    axes[1].legend(loc='best', ncol=max(1, len(param_names)//2)); axes[1].grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(Config.PLOTS_DIR / 'optimization_progress.png'); plt.close(fig)
    logging.info("Saved optimization progress plot.")

# ==================================
#            MAIN DRIVER
# ==================================
@use_named_args(Config.PARAM_SPACE)
def objective(**params):
    try:
        exp_data = read_experimental_data()
    except Exception:
        logging.critical("CRITICAL: Failed to load exp data. Aborting call."); return 1e10
    hf = Config.RESULTS_DIR / "optimization_history.json"
    try:
        with open(hf, 'r') as f: run_id = len(json.load(f)['iterations']) + 1
    except (FileNotFoundError, json.JSONDecodeError): run_id = 1
    total_error = run_single_evaluation(params, exp_data, run_id)
    plot_optimization_progress()
    return total_error

def main():
    logging.info("="*30 + "\n Starting DAMASK 3 Isothermal High-Temp Optimization \n" + "="*30)
    Config.setup()
    for f in [Config.MATERIAL_TEMPLATE, Config.GEOM_FILE, Config.LOAD_FILE, Config.EXP_DATA_FILE]:
        if not f.exists():
            logging.critical(f"CRITICAL: File not found: {f}. Exiting."); return
    try:
        read_experimental_data()
        logging.info("Experimental data read OK on initial check.")
    except Exception:
        logging.critical("Could not read exp data. Exiting.", exc_info=True); return
    logging.info("Config and files OK. Starting optimization...")
    start_time = datetime.now()
    res = gp_minimize(func=objective, dimensions=Config.PARAM_SPACE, n_calls=Config.MAX_ITER,
                      n_initial_points=max(1, Config.MAX_ITER // 2), verbose=True, n_jobs=Config.MAX_PARALLEL_JOBS)
    end_time = datetime.now()
    logging.info("="*30 + f"\n Optimization completed in: {end_time - start_time}\n" + "="*30)
    logging.info(f"Minimum objective value found: {res.fun:.6f}")
    logging.info("Best parameter set found:")
    best_params = {p.name: val for p, val in zip(Config.PARAM_SPACE, res.x)}
    for name, val in best_params.items(): logging.info(f"  {name}: {val:.6e}")
    with open(Config.RESULTS_DIR / 'best_parameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    logging.info(f"Best parameters saved to: {Config.RESULTS_DIR / 'best_parameters.json'}")
    try:
        plot_convergence(res)
        plt.savefig(Config.PLOTS_DIR / 'final_convergence.png')
        plt.close()
        logging.info(f"Final skopt convergence plot saved to: {Config.PLOTS_DIR / 'final_convergence.png'}")
    except Exception as e:
        logging.warning(f"Could not generate final skopt convergence plot: {e}")
    logging.info("Optimization process finished.")

if __name__ == "__main__":
    main()
