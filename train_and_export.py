"""Reproduces the final_pca.ipynb pipeline and exports fitted artifacts for serving."""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ARTIFACTS_DIR = "artifacts"

FEATURE_COLUMNS = [
    "Age", "Sex", "Height", "Weight", "QRS_Dur",
    "P-R_Int", "Q-T_Int", "T_Int", "P_Int", "QRS", "T", "P", "J", "Heart_Rate",
    "Q_Wave", "R_Wave", "S_Wave", "R'_Wave", "S'_Wave", "Int_Def", "Rag_R_Nom",
    "Diph_R_Nom", "Rag_P_Nom", "Diph_P_Nom", "Rag_T_Nom", "Diph_T_Nom",
    "DII00", "DII01", "DII02", "DII03", "DII04", "DII05", "DII06", "DII07", "DII08", "DII09", "DII10", "DII11",
    "DIII00", "DIII01", "DIII02", "DIII03", "DIII04", "DIII05", "DIII06", "DIII07", "DIII08", "DIII09", "DIII10", "DIII11",
    "AVR00", "AVR01", "AVR02", "AVR03", "AVR04", "AVR05", "AVR06", "AVR07", "AVR08", "AVR09", "AVR10", "AVR11",
    "AVL00", "AVL01", "AVL02", "AVL03", "AVL04", "AVL05", "AVL06", "AVL07", "AVL08", "AVL09", "AVL10", "AVL11",
    "AVF00", "AVF01", "AVF02", "AVF03", "AVF04", "AVF05", "AVF06", "AVF07", "AVF08", "AVF09", "AVF10", "AVF11",
    "V100", "V101", "V102", "V103", "V104", "V105", "V106", "V107", "V108", "V109", "V110", "V111",
    "V200", "V201", "V202", "V203", "V204", "V205", "V206", "V207", "V208", "V209", "V210", "V211",
    "V300", "V301", "V302", "V303", "V304", "V305", "V306", "V307", "V308", "V309", "V310", "V311",
    "V400", "V401", "V402", "V403", "V404", "V405", "V406", "V407", "V408", "V409", "V410", "V411",
    "V500", "V501", "V502", "V503", "V504", "V505", "V506", "V507", "V508", "V509", "V510", "V511",
    "V600", "V601", "V602", "V603", "V604", "V605", "V606", "V607", "V608", "V609", "V610", "V611",
    "JJ_Wave", "Amp_Q_Wave", "Amp_R_Wave", "Amp_S_Wave", "R_Prime_Wave", "S_Prime_Wave", "P_Wave", "T_Wave",
    "QRSA", "QRSTA", "DII170", "DII171", "DII172", "DII173", "DII174", "DII175", "DII176", "DII177", "DII178", "DII179",
    "DIII180", "DIII181", "DIII182", "DIII183", "DIII184", "DIII185", "DIII186", "DIII187", "DIII188", "DIII189",
    "AVR190", "AVR191", "AVR192", "AVR193", "AVR194", "AVR195", "AVR196", "AVR197", "AVR198", "AVR199",
    "AVL200", "AVL201", "AVL202", "AVL203", "AVL204", "AVL205", "AVL206", "AVL207", "AVL208", "AVL209",
    "AVF210", "AVF211", "AVF212", "AVF213", "AVF214", "AVF215", "AVF216", "AVF217", "AVF218", "AVF219",
    "V1220", "V1221", "V1222", "V1223", "V1224", "V1225", "V1226", "V1227", "V1228", "V1229",
    "V2230", "V2231", "V2232", "V2233", "V2234", "V2235", "V2236", "V2237", "V2238", "V2239",
    "V3240", "V3241", "V3242", "V3243", "V3244", "V3245", "V3246", "V3247", "V3248", "V3249",
    "V4250", "V4251", "V4252", "V4253", "V4254", "V4255", "V4256", "V4257", "V4258", "V4259",
    "V5260", "V5261", "V5262", "V5263", "V5264", "V5265", "V5266", "V5267", "V5268", "V5269",
    "V6270", "V6271", "V6272", "V6273", "V6274", "V6275", "V6276", "V6277", "V6278", "V6279",
]

CLASS_NAMES = [
    "Normal",
    "Ischemic changes (CAD)",
    "Old Anterior Myocardial Infraction",
    "Old Inferior Myocardial Infraction",
    "Sinus tachycardy",
    "Sinus bradycardy",
    "Ventricular Premature Contraction (PVC)",
    "Supraventricular Premature Contraction",
    "Left Boundle branch block",
    "Right boundle branch block",
    "1.Degree AtrioVentricular block",
    "2.Degree AV block",
    "3.Degree AV block",
    "Left Ventricule hypertrophy",
    "Atrial Fibrillation or Flutter",
    "Others",
]


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    df = pd.read_csv("Data/arrhythmia.csv", header=None)
    df = df.replace("?", np.nan)
    df.drop(columns=13, inplace=True)
    df.columns = FEATURE_COLUMNS + ["class"]

    df["Height"] = df["Height"].replace(608, 108)
    df["Height"] = df["Height"].replace(780, 180)

    target = df["class"]
    features = df.drop(columns="class")

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=1
    )

    imputer = SimpleImputer()
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    pca = PCA(0.98)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    model = RandomForestClassifier()
    model.fit(X_train_pca, y_train)

    y_pred = model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Test Accuracy: {accuracy}")
    print(f"Test F1 (weighted): {f1}")

    joblib.dump(imputer, f"{ARTIFACTS_DIR}/imputer.joblib")
    joblib.dump(scaler, f"{ARTIFACTS_DIR}/scaler.joblib")
    joblib.dump(pca, f"{ARTIFACTS_DIR}/pca.joblib")
    joblib.dump(model, f"{ARTIFACTS_DIR}/model.joblib")

    with open(f"{ARTIFACTS_DIR}/feature_schema.json", "w") as f:
        json.dump(
            {
                "feature_columns": FEATURE_COLUMNS,
                "class_names": CLASS_NAMES,
            },
            f,
            indent=2,
        )

    with open(f"{ARTIFACTS_DIR}/metrics.json", "w") as f:
        json.dump({"accuracy": accuracy, "f1_weighted": f1}, f, indent=2)


if __name__ == "__main__":
    main()
