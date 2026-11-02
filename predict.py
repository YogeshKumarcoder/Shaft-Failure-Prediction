import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Paths are relative to this file, not to the terminal folder
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "rf_mdet.joblib"
CONFIG_PATH = BASE / "models" / "config.json"

D_RANGE = (20, 150)          # mm, range the model was trained on
SYT_RANGE = (250, 1250)      # MPa
FOS_TRUSTED = (0.3, 2.5)     # Model is reliable only inside this FoS range

_model = None
_threshold = None


def _load():
    # Load once, reuse on every call
    global _model, _threshold

    if _model is None:
        _model = joblib.load(MODEL_PATH)

        with open(CONFIG_PATH, "r") as f:
            _threshold = json.load(f)["threshold"]

    return _model, _threshold


def formula_fos(d, M, T, Syt):
    """
    Calculate theoretical FoS, bending stress, and torsional shear stress.

    Parameters:
        d   : Shaft diameter in mm
        M   : Bending moment in N.mm
        T   : Torque in N.mm
        Syt : Yield strength in MPa

    Returns:
        fos, sigma, tau
    """

    if d <= 0:
        raise ValueError("Diameter d must be greater than zero.")

    if Syt <= 0:
        raise ValueError("Yield strength Syt must be greater than zero.")

    sigma = 32 * M / (np.pi * d**3)
    tau = 16 * T / (np.pi * d**3)

    equivalent_stress = np.sqrt(sigma**2 + 3 * tau**2)

    if equivalent_stress == 0:
        fos = np.inf
    else:
        fos = Syt / equivalent_stress

    return float(fos), float(sigma), float(tau)


def check_inputs(d, M, T, Syt):
    """
    Validate inputs.

    Invalid inputs raise ValueError.
    Out-of-range inputs generate warnings.
    """

    # Check that inputs are finite numbers
    values = {
        "d": d,
        "M": M,
        "T": T,
        "Syt": Syt
    }

    for name, value in values.items():
        if not np.isscalar(value) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")

    # Convert NumPy scalar values to ordinary Python floats
    d, M, T, Syt = map(float, (d, M, T, Syt))

    # Basic physical validation
    if d <= 0:
        raise ValueError("Diameter d must be greater than zero.")

    if Syt <= 0:
        raise ValueError("Yield strength Syt must be greater than zero.")

    if M < 0 or T < 0:
        raise ValueError("Bending moment M and torque T cannot be negative.")

    if M == 0 and T == 0:
        raise ValueError(
            "M and T cannot both be zero. "
            "There is no applied bending or torsional load."
        )

    warnings = []

    # Check diameter range
    if not D_RANGE[0] <= d <= D_RANGE[1]:
        warnings.append(
            f"Diameter d={d} mm is outside the training range "
            f"{D_RANGE} mm."
        )

    # Check yield strength range
    if not SYT_RANGE[0] <= Syt <= SYT_RANGE[1]:
        warnings.append(
            f"Yield strength Syt={Syt} MPa is outside the training range "
            f"{SYT_RANGE} MPa."
        )

    # Calculate theoretical FoS
    fos, sigma, tau = formula_fos(d, M, T, Syt)

    # Check whether the model's trusted FoS range is respected
    if not FOS_TRUSTED[0] <= fos <= FOS_TRUSTED[1]:
        warnings.append(
            f"Formula FoS={fos:.4f} is outside the trusted range "
            f"{FOS_TRUSTED}. The model is not reliable here."
        )

    # A very high FoS may indicate that moments were supplied in N.m
    # instead of N.mm, but it can also occur with genuinely low loads.
    if fos > 10:
        warnings.append(
            "Formula FoS is above 10. Check that M and T are in N.mm, "
            "not N.m. If your moments are in N.m, multiply them by 1000."
        )

    return warnings


def predict(d, M, T, Syt):
    """
    Predict FoS using the trained model and calculate theoretical FoS.

    Returns a dictionary containing:
        predicted_fos
        formula_fos
        sigma
        tau
        fail_risk
        warnings
    """

    # Validate inputs and collect warnings
    warnings = check_inputs(d, M, T, Syt)

    # Load the trained model and configured threshold
    model, threshold = _load()

    # Create a single-row DataFrame with the expected feature order
    X = pd.DataFrame(
        [[d, M, T, Syt]],
        columns=["d", "M", "T", "Syt"]
    )

    # Predict FoS
    predicted_fos = float(model.predict(X)[0])

    # Calculate theoretical values
    theoretical_fos, sigma, tau = formula_fos(d, M, T, Syt)

    # Estimate failure risk using the configured threshold
    fail_risk = predicted_fos < threshold

    return {
        "predicted_fos": predicted_fos,
        "formula_fos": theoretical_fos,
        "sigma": sigma,
        "tau": tau,
        "fail_risk": bool(fail_risk),
        "warnings": warnings
    }



if __name__ == "__main__":
    import sklearn

    print("Scikit-learn version:", sklearn.__version__)

    d = 50
    Syt = 400

    test_cases = [
        {"M": 6.0e6, "T": 3.0e6, "expected_fos": 0.75},
        {"M": 4.5e6, "T": 2.25e6, "expected_fos": 1.00},
        {"M": 3.0e6, "T": 1.5e6, "expected_fos": 1.50},
        {"M": 2.0e6, "T": 1.0e6, "expected_fos": 2.25},
    ]

    for i, case in enumerate(test_cases, start=1):
        M = case["M"]
        T = case["T"]
        expected_fos = case["expected_fos"]

        print(f"\n--- Test Case {i} ---")

        try:
            result = predict(d, M, T, Syt)

            print(f"M: {M:.2e} N.mm")
            print(f"T: {T:.2e} N.mm")
            print(f"Expected formula FoS: {expected_fos:.2f}")
            print(f"Actual formula FoS:   {result['formula_fos']:.4f}")
            print(f"Predicted FoS:        {result['predicted_fos']:.4f}")
            print(f"Failure risk:         {result['fail_risk']}")
            print(f"Warnings:             {result['warnings']}")

        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")
