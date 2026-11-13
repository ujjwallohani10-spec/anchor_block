"""
===============
Penstock anchor block stability check (preliminary design).

Based on IS 11639 (Part 2):1995 and IS 1893:1984.

Checks, for both the XY and ZY planes:
  - Overturning
  - Sliding
  - Eccentricity (middle-third rule)
  - Bearing pressure
under static and seismic conditions, for pipe expansion and contraction.

Expansion / contraction
    Friction at the pipe supports / expansion joint is applied as a
    horizontal force of magnitude `joint_friction_kN` at pipe-axis height.
      - Expansion  : friction acts downstream (adds to the earth-pressure load)
      - Contraction: friction acts upstream   (opposes the earth-pressure load)
    With joint_friction_kN = 0 both cases give identical results.

Sign convention
    X positive = upstream, Y positive = downward, Z = along the block width.

All input values below are ILLUSTRATIVE EXAMPLES, not data from a real project.

DISCLAIMER: for learning and preliminary checks only. Results must be
verified independently before use in any design for construction.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# -----------------------------------------------------------------------------
# 1. INPUT DATA STRUCTURES
# -----------------------------------------------------------------------------

@dataclass
class MaterialWeights:
    """Unit weights (kN/m3)."""
    gamma_c: float = 25.0       # Concrete
    gamma_s: float = 20.0       # Soil
    gamma_w: float = 9.81       # Water
    gamma_p: float = 77.0085    # Steel pipe


@dataclass
class SoilProperties:
    phi: float = 30.0            # Internal friction angle, deg
    q_na: float = 200.0          # Net allowable bearing capacity, kN/m2
    mu: float = 0.55             # Friction coefficient at block base
    wall_friction: float = 20.0  # Wall friction angle delta, deg
    slope_angle: float = 0.0     # Backfill slope i, deg
    wall_angle: float = 0.0      # Wall face angle with vertical, deg


@dataclass
class SeismicProperties:
    beta: float = 1.00           # Soil-foundation factor
    I: float = 1.50              # Importance factor
    alpha_o: float = 0.08        # Basic horizontal seismic coefficient
    alpha_h: float = field(init=False)
    alpha_v: float = field(init=False)

    def __post_init__(self):
        self.alpha_h = self.beta * self.I * self.alpha_o
        self.alpha_v = self.alpha_h * 0.5


@dataclass
class SafetyFactors:
    """Minimum required factors of safety. Confirm against the code you design to."""
    overturning_static: float = 2.0
    overturning_seismic: float = 1.1
    sliding_static: float = 1.5
    sliding_seismic: float = 1.2


@dataclass
class PipeData:
    Q: float = 6.0                   # Discharge, m3/s
    d: float = 1600.0                # Outer diameter, mm
    t: float = 16.0                  # Wall thickness, mm
    alpha_us: float = 5.0            # Upstream inclination from vertical, deg
    beta_ds: float = 3.0             # Downstream inclination from vertical, deg
    H_total: float = 45.0            # Design head (static + surge), m
    joint_friction_kN: float = 20.0  # Friction at supports / expansion joint, kN

    @property
    def r_outer(self) -> float:
        return self.d / 2000.0

    @property
    def r_inner(self) -> float:
        return (self.d / 2.0 - self.t) / 1000.0

    @property
    def A_inner(self) -> float:
        return math.pi * self.r_inner ** 2

    @property
    def A_outer(self) -> float:
        """Outer area, used for the hydrostatic bend force (conservative)."""
        return math.pi * self.r_outer ** 2

    @property
    def A_pipe_section(self) -> float:
        """Annular steel area, m2."""
        return math.pi * (self.r_outer ** 2 - self.r_inner ** 2)

    @property
    def V_flow(self) -> float:
        return self.Q / self.A_inner

    @property
    def xi(self) -> float:
        """Net bend angle at the anchor block, deg."""
        return abs(self.alpha_us - self.beta_ds)

    @property
    def bisector_angle(self) -> float:
        return (self.alpha_us + self.beta_ds) / 2.0


@dataclass
class BlockDimensions:
    L_us: float = 1.7           # Block / pipe length, upstream side, m
    L_ds: float = 1.7           # Block / pipe length, downstream side, m
    w: float = 3.2              # Block width (Z direction), m
    H1: float = 3.0             # Block height at upstream face, m
    H2: float = 2.8             # Block height at downstream face, m
    h1: float = 3.65            # Total depth at upstream face (block + cover), m
    h_backfill: float = 0.60    # Backfill height over block top, m
    h_joint: float = 2.0        # Pipe axis height above base, m
    V_block: Optional[float] = None  # Gross volume, m3 (None = estimated from H1, H2)
    CG_X: Optional[float] = None     # CG from downstream face, m (None = estimated)

    @property
    def L_total(self) -> float:
        return self.L_us + self.L_ds

    @property
    def A_base(self) -> float:
        return self.L_total * self.w

    @property
    def volume(self) -> float:
        if self.V_block is not None:
            return self.V_block
        return self.L_total * self.w * (self.H1 + self.H2) / 2.0

    @property
    def cg_x(self) -> float:
        """Centroid from the downstream face for linearly varying height."""
        if self.CG_X is not None:
            return self.CG_X
        return self.L_total * (self.H2 + 2.0 * self.H1) / (3.0 * (self.H1 + self.H2))

    @property
    def cg_z(self) -> float:
        return self.w / 2.0


# -----------------------------------------------------------------------------
# 2. CALCULATOR ENGINE
# -----------------------------------------------------------------------------

class AnchorBlockCalculator:
    """
    Usage
    -----
        calc   = AnchorBlockCalculator(PipeData(), BlockDimensions())
        result = calc.run_all()
        print(result["summary_df"].to_string())
    """

    DEG = math.pi / 180.0

    def __init__(
        self,
        pipe: PipeData,
        block: BlockDimensions,
        materials: Optional[MaterialWeights] = None,
        soil: Optional[SoilProperties] = None,
        seismic: Optional[SeismicProperties] = None,
        safety: Optional[SafetyFactors] = None,
    ):
        self.pipe = pipe
        self.block = block
        self.mat = materials or MaterialWeights()
        self.soil = soil or SoilProperties()
        self.seis = seismic or SeismicProperties()
        self.sf = safety or SafetyFactors()

    # -- A. Volumes and weights -------------------------------------------------

    def _volumes_and_weights(self) -> dict:
        p, b, m = self.pipe, self.block, self.mat

        V_block = b.volume
        V_pipe = math.pi * p.r_outer ** 2 * b.L_total
        V_net = V_block - V_pipe
        W_B = V_net * m.gamma_c

        W_p_lin = p.A_pipe_section * m.gamma_p     # steel, kN/m
        W_w_lin = p.A_inner * m.gamma_w            # water, kN/m
        W_lin = W_p_lin + W_w_lin

        F_overburden = b.h_backfill * b.L_total * b.w * m.gamma_s

        return {
            "V_block_m3": round(V_block, 4),
            "V_pipe_m3": round(V_pipe, 4),
            "V_net_m3": round(V_net, 4),
            "W_B_kN": round(W_B, 4),
            "W_pipe_per_m_kN": round(W_p_lin, 4),
            "W_water_per_m_kN": round(W_w_lin, 4),
            "W_total_per_m_kN": round(W_lin, 4),
            "F_overburden_kN": round(F_overburden, 4),
        }

    # -- B. Earth pressure coefficients ------------------------------------------

    def _earth_pressure_coefficients(self) -> dict:
        """
        Mononobe-Okabe formulation. With lambda = 0 it reduces to the Coulomb
        coefficient with wall friction (Ka ~ 0.297 for phi = 30, delta = 20 deg).
        """
        phi = self.soil.phi * self.DEG
        delta = self.soil.wall_friction * self.DEG
        alpha = self.soil.wall_angle * self.DEG
        i = self.soil.slope_angle * self.DEG
        ah = self.seis.alpha_h
        av = self.seis.alpha_v

        Ka = self._kaeq(phi, delta, alpha, i, lam=0.0)

        lam_pos = math.atan(ah / (1.0 + av))
        lam_neg = math.atan(ah / (1.0 - av))

        Kaeq_pos = self._kaeq(phi, delta, alpha, i, lam_pos)
        Kaeq_neg = self._kaeq(phi, delta, alpha, i, lam_neg)

        valid = [k for k in (Kaeq_pos, Kaeq_neg) if k is not None]
        if not valid:
            raise ValueError(
                "Seismic angle exceeds the soil friction angle: "
                "Mononobe-Okabe coefficient is undefined. Check phi and seismic inputs."
            )
        Kaeq = max(valid)  # most critical of the two vertical-acceleration cases

        return {
            "Ka": round(Ka, 5),
            "lambda_pos_deg": round(math.degrees(lam_pos), 3),
            "lambda_neg_deg": round(math.degrees(lam_neg), 3),
            "Kaeq_pos": round(Kaeq_pos, 5) if Kaeq_pos is not None else None,
            "Kaeq_neg": round(Kaeq_neg, 5) if Kaeq_neg is not None else None,
            "Kaeq": round(Kaeq, 5),
        }

    @staticmethod
    def _kaeq(phi, delta, alpha, i, lam):
        """
        Mononobe-Okabe active earth pressure coefficient (IS 1893:1984 form):

            cos^2(phi - lam - alpha)
          ------------------------------------------------------------------
          cos(lam) cos^2(alpha) cos(delta + alpha + lam) [1 + sqrt(S)]^2

            S = sin(phi + delta) sin(phi - lam - i)
                / [cos(delta + alpha + lam) cos(i - alpha)]

        Returns None where the coefficient is undefined (phi - lam <= i).
        """
        if phi - lam <= i:
            return None
        try:
            num = math.cos(phi - lam - alpha) ** 2
            S = (math.sin(phi + delta) * math.sin(phi - lam - i)
                 / (math.cos(delta + alpha + lam) * math.cos(i - alpha)))
            if S < 0:
                return None
            denom = (math.cos(lam) * math.cos(alpha) ** 2
                     * math.cos(delta + alpha + lam) * (1.0 + math.sqrt(S)) ** 2)
            return num / denom if denom != 0 else None
        except (ValueError, ZeroDivisionError):
            return None

    # -- C. Force resolution -----------------------------------------------------

    def _resolve_forces(self, ep: dict, vw: dict) -> dict:
        """
        Forces resolved into X and Y components:
          F1u, F1d     weight of pipe + water, upstream / downstream reach
          F3           hydrostatic force at the bend
          F8           dynamic force from the change of flow direction
          F10          active soil pressure on the upstream face
          F_overburden backfill above the block
          W_B          block self-weight
        """
        p, b, m = self.pipe, self.block, self.mat

        a = p.alpha_us * self.DEG
        be = p.beta_ds * self.DEG
        xi = p.xi * self.DEG
        g = p.bisector_angle * self.DEG

        W_lin = vw["W_total_per_m_kN"]
        W_B = vw["W_B_kN"]
        F_ob = vw["F_overburden_kN"]

        F1u = W_lin * b.L_us
        F1d = W_lin * b.L_ds
        F1u_x, F1u_y = F1u * math.sin(a), F1u * math.cos(a)
        F1d_x, F1d_y = F1d * math.sin(be), F1d * math.cos(be)

        P_h = m.gamma_w * p.H_total
        F3 = 2.0 * P_h * p.A_outer * math.sin(xi / 2.0)
        F3_x, F3_y = -F3 * math.sin(g), -F3 * math.cos(g)

        rho_w = 1000.0
        F8 = rho_w * p.Q * p.V_flow * 2.0 * math.sin(xi / 2.0) / 1000.0
        F8_x, F8_y = -F8 * math.sin(g), -F8 * math.cos(g)

        F10 = 0.5 * ep["Ka"] * m.gamma_s * b.h1 ** 2 * b.w
        F10_x = -F10

        sum_Hx_bend = F1u_x + F1d_x + F3_x + F8_x
        sum_Hx = sum_Hx_bend + F10_x
        sum_Vy_bend = F1u_y + F1d_y + F3_y + F8_y
        sum_V = sum_Vy_bend + W_B + F_ob

        return {
            "F1u_kN": round(F1u, 4), "F1d_kN": round(F1d, 4),
            "F1u_x_kN": round(F1u_x, 4), "F1u_y_kN": round(F1u_y, 4),
            "F1d_x_kN": round(F1d_x, 4), "F1d_y_kN": round(F1d_y, 4),
            "F3_kN": round(F3, 4), "F3_x_kN": round(F3_x, 4), "F3_y_kN": round(F3_y, 4),
            "F8_kN": round(F8, 4), "F8_x_kN": round(F8_x, 4), "F8_y_kN": round(F8_y, 4),
            "F10_kN": round(F10, 4), "F10_x_kN": round(F10_x, 4),
            "F_overburden_kN": round(F_ob, 4),
            "W_B_kN": round(W_B, 4),
            "sum_Hx_bend_kN": round(sum_Hx_bend, 4),
            "sum_Vy_bend_kN": round(sum_Vy_bend, 4),
            "sum_Hx_kN": round(sum_Hx, 4),
            "sum_Hz_kN": 0.0,
            "sum_V_kN": round(sum_V, 4),
        }

    # -- D. Stability check for one plane / case ---------------------------------

    def _check_plane(self, forces: dict, ep: dict, plane: str,
                     seismic: bool, case: str) -> dict:
        """
        XY plane: block tips about the downstream toe.
        ZY plane: block tips sideways about a lateral edge. As in the reference
        method, the same earth-pressure overturning moment is used for both
        planes (a conservative simplification).
        """
        b, s, seis, sf = self.block, self.soil, self.seis, self.sf
        xy = plane == "XY"

        W_B = forces["W_B_kN"]
        F_ob = forces["F_overburden_kN"]
        sum_Vy_bend = forces["sum_Vy_bend_kN"]
        sum_Hz = forces["sum_Hz_kN"]

        # Friction at supports / joint: expansion acts downstream, contraction upstream
        F_fr = self.pipe.joint_friction_kN
        Hx_bend = forces["sum_Hx_bend_kN"] + (-F_fr if case == "expansion" else F_fr)

        # Earth pressure on the upstream face (seismic uses Kaeq)
        K = ep["Kaeq"] if seismic else ep["Ka"]
        F10 = 0.5 * K * self.mat.gamma_s * b.h1 ** 2 * b.w

        # Seismic effects on the block mass
        W_B_eff = W_B * (1.0 - seis.alpha_v) if seismic else W_B
        seismic_H = seis.alpha_h * W_B if seismic else 0.0

        sum_V = sum_Vy_bend + W_B_eff + F_ob

        # Lever arms about the toe
        arm_WB = b.cg_x if xy else b.cg_z
        arm_Vy = b.L_ds if xy else b.w / 2.0
        arm_F10 = b.h1 / 3.0
        arm_seis = b.H1 / 2.0

        # Moments
        M_R = W_B_eff * arm_WB + F_ob * arm_WB + sum_Vy_bend * arm_Vy
        M_O = F10 * arm_F10
        if xy:
            M_bend = Hx_bend * b.h_joint       # upstream force resists, downstream overturns
            if M_bend >= 0:
                M_R += M_bend
            else:
                M_O += -M_bend
        if seismic:
            M_O += seismic_H * arm_seis
        M_net = M_R - M_O

        # Overturning
        FOS_OT = M_R / M_O if M_O > 0 else float("inf")
        OT_req = sf.overturning_seismic if seismic else sf.overturning_static
        status_OT = "SAFE" if FOS_OT >= OT_req else "FAILED"

        # Sliding: driving force magnitude plus the seismic force (conservative)
        H_static = (Hx_bend - F10) if xy else sum_Hz
        H_drive = abs(H_static) + seismic_H
        FOS_SL = s.mu * abs(sum_V) / H_drive if H_drive > 1e-6 else float("inf")
        SL_req = sf.sliding_seismic if seismic else sf.sliding_static
        status_SL = "SAFE" if FOS_SL >= SL_req else "FAILED"

        # Eccentricity (middle-third rule)
        L_dim = b.L_total if xy else b.w
        e_perm = L_dim / 6.0
        e = L_dim / 2.0 - M_net / abs(sum_V) if abs(sum_V) > 1e-6 else L_dim / 2.0
        status_e = "SAFE" if abs(e) <= e_perm else "FAILED"

        # Bearing pressure
        P_avg = abs(sum_V) / b.A_base
        P_max = P_avg * (1.0 + 6.0 * abs(e) / L_dim)
        P_min = P_avg * (1.0 - 6.0 * abs(e) / L_dim)
        status_BP = "SAFE" if P_max <= s.q_na else "FAILED"

        overall = "SAFE" if all(
            st == "SAFE" for st in (status_OT, status_SL, status_e, status_BP)
        ) else "FAILED"

        return {
            "plane": plane,
            "case": case,
            "seismic": seismic,
            "sum_V_kN": round(sum_V, 3),
            "sum_H_kN": round(-H_drive, 3),
            "M_R_kNm": round(M_R, 3),
            "M_O_kNm": round(M_O, 3),
            "M_net_kNm": round(M_net, 3),
            "FOS_OT": round(FOS_OT, 3) if FOS_OT != float("inf") else 999.0,
            "FOS_OT_req": OT_req,
            "status_OT": status_OT,
            "FOS_SL": round(FOS_SL, 3) if FOS_SL != float("inf") else 999.0,
            "FOS_SL_req": SL_req,
            "status_SL": status_SL,
            "e_m": round(e, 4),
            "e_perm_m": round(e_perm, 4),
            "status_e": status_e,
            "P_max_kNm2": round(P_max, 3),
            "P_min_kNm2": round(P_min, 3),
            "BP_allow_kNm2": s.q_na,
            "status_BP": status_BP,
            "overall": overall,
        }

    # -- E. Public API -----------------------------------------------------------

    def run_all(self) -> dict:
        """Run every plane / case / condition combination (8 checks)."""
        vw = self._volumes_and_weights()
        ep = self._earth_pressure_coefficients()
        forces = self._resolve_forces(ep, vw)

        checks = {}
        for seismic in (False, True):
            for case in ("expansion", "contraction"):
                for plane in ("XY", "ZY"):
                    key = f"{'seismic' if seismic else 'static'}_{case}_{plane}"
                    checks[key] = self._check_plane(forces, ep, plane, seismic, case)

        return {
            "volumes_and_weights": vw,
            "earth_pressure_coefficients": ep,
            "force_resolution": forces,
            "stability_checks": checks,
            "summary_df": self._build_dataframe(checks),
        }

    @staticmethod
    def _build_dataframe(checks: dict) -> pd.DataFrame:
        rows = []
        for c in checks.values():
            rows.append({
                "Condition": "Seismic" if c["seismic"] else "Static",
                "Case": c["case"].title(),
                "Plane": c["plane"],
                "SumV (kN)": c["sum_V_kN"],
                "SumH (kN)": c["sum_H_kN"],
                "M_R (kNm)": c["M_R_kNm"],
                "M_O (kNm)": c["M_O_kNm"],
                "e (m)": c["e_m"],
                "e_perm (m)": c["e_perm_m"],
                "FOS_OT": c["FOS_OT"],
                "OT_req": c["FOS_OT_req"],
                "OT Status": c["status_OT"],
                "FOS_SL": c["FOS_SL"],
                "SL_req": c["FOS_SL_req"],
                "SL Status": c["status_SL"],
                "P_max (kN/m2)": c["P_max_kNm2"],
                "P_min (kN/m2)": c["P_min_kNm2"],
                "BP_allow (kN/m2)": c["BP_allow_kNm2"],
                "BP Status": c["status_BP"],
                "e Status": c["status_e"],
                "OVERALL": c["overall"],
            })
        df = pd.DataFrame(rows)
        df.sort_values(["Condition", "Case", "Plane"], inplace=True, ignore_index=True)
        return df


# -----------------------------------------------------------------------------
# 3. CONVENIENCE FACTORY
# -----------------------------------------------------------------------------

def build_from_dict(params: dict) -> AnchorBlockCalculator:
    """
    Build a calculator from a plain dictionary (e.g. loaded from JSON).
    Optional top-level keys: "pipe", "block", "materials", "soil", "seismic",
    "safety". Each maps to the keyword arguments of the matching dataclass.
    """
    return AnchorBlockCalculator(
        PipeData(**params.get("pipe", {})),
        BlockDimensions(**params.get("block", {})),
        MaterialWeights(**params.get("materials", {})),
        SoilProperties(**params.get("soil", {})),
        SeismicProperties(**params.get("seismic", {})),
        SafetyFactors(**params.get("safety", {})),
    )


# -----------------------------------------------------------------------------
# 4. DEMO
# -----------------------------------------------------------------------------

def main():
    """Run the analysis with the illustrative default inputs and print results."""
    calc = AnchorBlockCalculator(PipeData(), BlockDimensions())
    result = calc.run_all()

    vw = result["volumes_and_weights"]
    ep = result["earth_pressure_coefficients"]
    f = result["force_resolution"]
    df = result["summary_df"]
    line = "=" * 78

    print(line)
    print("  ANCHOR BLOCK STABILITY CHECK - EXAMPLE INPUTS")
    print(line)
    print(f"\n  Net block volume   : {vw['V_net_m3']:.2f} m3")
    print(f"  Block weight       : {vw['W_B_kN']:.1f} kN")
    print(f"  Overburden         : {vw['F_overburden_kN']:.1f} kN")
    print(f"  Ka / Kaeq          : {ep['Ka']:.4f} / {ep['Kaeq']:.4f}")
    print(f"  Hydrostatic F3     : {f['F3_kN']:.2f} kN")
    print(f"  Dynamic F8         : {f['F8_kN']:.2f} kN")
    print(f"  Earth pressure F10 : {f['F10_kN']:.2f} kN")

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    cols = ["Condition", "Case", "Plane", "FOS_OT", "OT Status", "FOS_SL",
            "SL Status", "P_max (kN/m2)", "BP Status", "e (m)", "e_perm (m)",
            "e Status", "OVERALL"]
    print(f"\n{line}\n  SUMMARY\n{line}")
    print(df[cols].to_string(index=False))
    print(line)
    return result


if __name__ == "__main__":
    main()