from pathlib import Path
import math

from flask import Flask, render_template, request
import joblib
import pandas as pd


app = Flask(__name__)

# Resolve the saved artifacts relative to this file. This allows the app to be
# started from any working directory without breaking the joblib paths.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "svm_model.pkl"
SCALER_PATH = BASE_DIR / "svm_scaler.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "svm_feature_columns.pkl"


def load_artifact(path, description):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path.name}. "
            "Place it in the same folder as app.py."
        )
    return joblib.load(path)


model = load_artifact(MODEL_PATH, "trained SVM model")
scaler = load_artifact(SCALER_PATH, "fitted scaler")
feature_columns = list(
    load_artifact(FEATURE_COLUMNS_PATH, "saved feature-column list")
)

# The source dataset kept some names capitalized (for example Scholarship,
# Diabetes and Alcoholism). Resolve ordinary field names to the model's exact
# saved spelling so reindex() does not silently replace submitted values with 0.
FEATURE_NAME_LOOKUP = {
    column.casefold(): column
    for column in feature_columns
    if not column.startswith(("appt_weekday_", "neighbourhood_"))
}

# These are the only columns scaled in the training notebook.
NUMERICAL_COLUMNS = ["age", "days_until_appointment"]

missing_numerical_columns = set(NUMERICAL_COLUMNS) - set(feature_columns)
if missing_numerical_columns:
    raise ValueError(
        "The saved feature list is incompatible with the app. Missing: "
        + ", ".join(sorted(missing_numerical_columns))
    )

# get_dummies(drop_first=True) removed the first category from each group.
# For this dataset those reference categories are Friday and AEROPORTO. When a
# reference category is selected, all dummy columns in that group correctly
# remain zero.
BASELINE_WEEKDAY = "Friday"
BASELINE_NEIGHBOURHOOD = "AEROPORTO"

weekday_options = sorted(
    {BASELINE_WEEKDAY}
    | {
        column.removeprefix("appt_weekday_")
        for column in feature_columns
        if column.startswith("appt_weekday_")
    }
)

neighbourhood_options = sorted(
    {BASELINE_NEIGHBOURHOOD}
    | {
        column.removeprefix("neighbourhood_")
        for column in feature_columns
        if column.startswith("neighbourhood_")
    }
)


def model_feature_name(name):
    """Return a base feature's exact name from the saved training columns."""
    try:
        return FEATURE_NAME_LOOKUP[name.casefold()]
    except KeyError as exc:
        raise ValueError(
            f"The saved model does not contain the required '{name}' feature."
        ) from exc


def parse_integer(form, field, minimum=None, maximum=None, default=None):
    """Read and validate an integer form field."""
    raw_value = form.get(field, default)
    if raw_value in (None, ""):
        raise ValueError(f"{field.replace('_', ' ').title()} is required.")

    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field.replace('_', ' ').title()} must be a whole number."
        ) from exc

    if minimum is not None and value < minimum:
        raise ValueError(
            f"{field.replace('_', ' ').title()} cannot be below {minimum}."
        )
    if maximum is not None and value > maximum:
        raise ValueError(
            f"{field.replace('_', ' ').title()} cannot exceed {maximum}."
        )
    return value


def parse_binary(form, field):
    """Handle either 0/1 selects or ordinary HTML checkboxes."""
    raw_value = str(form.get(field, "0")).strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return 1
    if raw_value in {"0", "false", "no", "off", ""}:
        return 0
    raise ValueError(f"Invalid value supplied for {field.replace('_', ' ')}.")


def build_input_row(form):
    """Transform one submitted appointment into the model's feature layout."""
    gender = str(form.get("gender", "")).strip().upper()
    if gender not in {"F", "M"}:
        raise ValueError("Gender must be F or M.")

    weekday = str(form.get("appt_weekday", "")).strip()
    if weekday not in weekday_options:
        raise ValueError("Please select a valid appointment weekday.")

    neighbourhood = str(form.get("neighbourhood", "")).strip()
    if neighbourhood not in neighbourhood_options:
        raise ValueError("Please select a valid neighbourhood.")

    row = {
        model_feature_name("age"): parse_integer(
            form, "age", minimum=0, maximum=120
        ),
        model_feature_name("gender"): 1 if gender == "M" else 0,
        model_feature_name("scholarship"): parse_binary(form, "scholarship"),
        model_feature_name("hypertension"): parse_binary(form, "hypertension"),
        model_feature_name("diabetes"): parse_binary(form, "diabetes"),
        model_feature_name("alcoholism"): parse_binary(form, "alcoholism"),
        model_feature_name("handicap"): parse_integer(
            form, "handicap", minimum=0, maximum=4, default=0
        ),
        model_feature_name("sms_received"): parse_binary(form, "sms_received"),
        model_feature_name("days_until_appointment"): parse_integer(
            form, "days_until_appointment", minimum=0
        ),
    }

    for column in feature_columns:
        if column.startswith("appt_weekday_"):
            row[column] = int(column == f"appt_weekday_{weekday}")
        elif column.startswith("neighbourhood_"):
            row[column] = int(column == f"neighbourhood_{neighbourhood}")

    # reindex both creates missing dummy columns as zero and guarantees the
    # exact feature order used when the SVM was trained.
    input_df = pd.DataFrame([row]).reindex(
        columns=feature_columns, fill_value=0
    )
    input_df[NUMERICAL_COLUMNS] = scaler.transform(
        input_df[NUMERICAL_COLUMNS]
    )

    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is not None and input_df.shape[1] != expected_features:
        raise ValueError(
            f"Model expects {expected_features} features, but the app produced "
            f"{input_df.shape[1]}. Re-export the matching feature-column file."
        )

    return input_df


def sigmoid(value):
    """Convert an SVM margin into a bounded score without numerical overflow."""
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def predict_no_show(input_df):
    """Return the binary prediction and the best available no-show score."""
    prediction = int(model.predict(input_df)[0])
    classes = list(getattr(model, "classes_", [0, 1]))

    if 1 not in classes:
        raise ValueError("The loaded model has no class labelled 1 (no-show).")

    # This branch is used automatically if the SVC is later retrained with
    # probability=True or replaced by another probabilistic classifier.
    if hasattr(model, "predict_proba"):
        no_show_index = classes.index(1)
        probability = float(model.predict_proba(input_df)[0][no_show_index])
        return prediction, probability, True

    # The current final SVC has probability=False. Its decision_function is a
    # signed distance from the classification boundary, not a calibrated
    # probability. Applying a sigmoid provides a useful 0-100 relative risk
    # score while preserving 50% as the SVM decision boundary.
    if hasattr(model, "decision_function"):
        margin = float(model.decision_function(input_df)[0])
        if len(classes) == 2 and classes[1] != 1:
            margin = -margin
        return prediction, sigmoid(margin), False

    return prediction, None, False


def prediction_details(prediction):
    """Map the SVM's 0/1 output to user-facing result text."""
    if prediction == 1:
        return (
            "No-show predicted",
            "High",
            "red",
            "The patient is likely to miss the appointment. Send a reminder.",
        )
    return (
        "Attendance predicted",
        "Low",
        "green",
        "The patient is likely to attend the appointment.",
    )


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "weekday_options": weekday_options,
        "neighbourhood_options": neighbourhood_options,
        "form_data": request.form if request.method == "POST" else {},
    }

    if request.method == "POST":
        try:
            input_df = build_input_row(request.form)
            prediction, score, is_probability = predict_no_show(input_df)
            result, risk_level, color, message = prediction_details(prediction)

            context.update(
                {
                    "prediction": prediction,
                    "result": result,
                    "risk_level": risk_level,
                    "color": color,
                    "message": message,
                    # Keep `proba` for compatibility with the existing template.
                    # For the current SVC it contains a relative risk score.
                    "proba": round(score * 100, 1) if score is not None else None,
                    "score_label": (
                        "No-show probability"
                        if is_probability
                        else "No-show risk score"
                    ),
                    "is_probability": is_probability,
                    "score_note": (
                        None
                        if is_probability
                        else "This is a relative score from the SVM decision "
                        "margin, not a calibrated probability."
                    ),
                }
            )
        except ValueError as exc:
            context["error"] = str(exc)
            return render_template("index.html", **context), 400

    return render_template("index.html", **context)


if __name__ == "__main__":
    # Keep debug mode off by default so the development server does not expose
    # its interactive debugger. Enable it explicitly through Flask when needed.
    app.run(debug=False)
