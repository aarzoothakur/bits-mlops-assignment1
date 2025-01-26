import pytest
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


@pytest.fixture
def load_data():
    # Load the CSV data
    data = pd.read_csv('./data/loan_data.csv')
    # Rename columns (bulk renaming)
    data.rename(columns={
        'SK_ID_CURR': 'id',
        'TARGET': 'target',
        'NAME_CONTRACT_TYPE': 'contract_type',
        'CODE_GENDER': 'gender',
        'FLAG_OWN_CAR': 'own_car',
        'FLAG_OWN_REALTY': 'own_realty',
        'CNT_CHILDREN': 'children_count',
        'AMT_INCOME_TOTAL': 'total_income',
        'AMT_CREDIT': 'credit_amount',
        'AMT_ANNUITY': 'annuity_amount',
        'AMT_GOODS_PRICE': 'goods_price',
        'NAME_TYPE_SUITE': 'suite_type',
        'NAME_INCOME_TYPE': 'income_type',
        'NAME_EDUCATION_TYPE': 'education_type',
        'NAME_FAMILY_STATUS': 'family_status',
        'NAME_HOUSING_TYPE': 'housing_type',
        'REGION_POPULATION_RELATIVE': 'population_relative',
        'DAYS_BIRTH': 'days_birth',
        'DAYS_EMPLOYED': 'days_employed',
        'DAYS_REGISTRATION': 'days_registration',
        'DAYS_ID_PUBLISH': 'days_id_publish',
        'OWN_CAR_AGE': 'car_age',
        'FLAG_MOBIL': 'has_mobile',
        'FLAG_EMP_PHONE': 'has_emp_phone',
        'FLAG_WORK_PHONE': 'has_work_phone',
        'FLAG_CONT_MOBILE': 'mobile_contactable',
        'FLAG_PHONE': 'has_phone',
        'FLAG_EMAIL': 'has_email',
        'OCCUPATION_TYPE': 'occupation_type',
        'CNT_FAM_MEMBERS': 'family_members_count',
        'REG_REGION_NOT_LIVE_REGION': 'region_mismatch_live',
        'REG_REGION_NOT_WORK_REGION': 'region_mismatch_work',
        'LIVE_REGION_NOT_WORK_REGION': 'live_work_region_mismatch',
        'REG_CITY_NOT_LIVE_CITY': 'city_mismatch_live',
        'REG_CITY_NOT_WORK_CITY': 'city_mismatch_work',
        'LIVE_CITY_NOT_WORK_CITY': 'live_work_city_mismatch',
        'ORGANIZATION_TYPE': 'organization_type',
        'EXT_SOURCE_1': 'ext_source_1',
        'EXT_SOURCE_2': 'ext_source_2',
        'EXT_SOURCE_3': 'ext_source_3',
        'APARTMENTS_AVG': 'avg_apartments',
        'BASEMENTAREA_AVG': 'avg_basement_area',
        'YEARS_BEGINEXPLUATATION_AVG': 'avg_years_begin_expl',
        'YEARS_BUILD_AVG': 'avg_years_build',
        'COMMONAREA_AVG': 'avg_common_area',
        'ELEVATORS_AVG': 'avg_elevators',
        'ENTRANCES_AVG': 'avg_entrances',
        'FLOORSMAX_AVG': 'avg_max_floors',
        'FLOORSMIN_AVG': 'avg_min_floors',
        'LANDAREA_AVG': 'avg_land_area',
        'LIVINGAPARTMENTS_AVG': 'avg_living_apartments',
        'LIVINGAREA_AVG': 'avg_living_area',
        'NONLIVINGAPARTMENTS_AVG': 'avg_nonliving_apartments',
        'NONLIVINGAREA_AVG': 'avg_nonliving_area',
        'APARTMENTS_MODE': 'mode_apartments',
        'BASEMENTAREA_MODE': 'mode_basement_area',
        'YEARS_BEGINEXPLUATATION_MODE': 'mode_years_begin_expl',
        'YEARS_BUILD_MODE': 'mode_years_build',
        'COMMONAREA_MODE': 'mode_common_area',
        'ELEVATORS_MODE': 'mode_elevators',
        'ENTRANCES_MODE': 'mode_entrances',
        'FLOORSMAX_MODE': 'mode_max_floors',
        'FLOORSMIN_MODE': 'mode_min_floors',
        'LANDAREA_MODE': 'mode_land_area',
        'LIVINGAPARTMENTS_MODE': 'mode_living_apartments',
        'LIVINGAREA_MODE': 'mode_living_area',
        'NONLIVINGAPARTMENTS_MODE': 'mode_nonliving_apartments',
        'NONLIVINGAREA_MODE': 'mode_nonliving_area',
        'APARTMENTS_MEDI': 'medi_apartments',
        'BASEMENTAREA_MEDI': 'medi_basement_area',
        'YEARS_BEGINEXPLUATATION_MEDI': 'medi_years_begin_expl',
        'YEARS_BUILD_MEDI': 'medi_years_build',
        'COMMONAREA_MEDI': 'medi_common_area',
        'ELEVATORS_MEDI': 'medi_elevators',
        'ENTRANCES_MEDI': 'medi_entrances',
        'FLOORSMAX_MEDI': 'medi_max_floors',
        'FLOORSMIN_MEDI': 'medi_min_floors',
        'LANDAREA_MEDI': 'medi_land_area',
        'LIVINGAPARTMENTS_MEDI': 'medi_living_apartments',
        'LIVINGAREA_MEDI': 'medi_living_area',
        'NONLIVINGAPARTMENTS_MEDI': 'medi_nonliving_apartments',
        'NONLIVINGAREA_MEDI': 'medi_nonliving_area',
        'FONDKAPREMONT_MODE': 'fondk_premont_mode',
        'HOUSETYPE_MODE': 'house_type_mode',
        'TOTALAREA_MODE': 'total_area_mode',
        'WALLSMATERIAL_MODE': 'walls_material_mode',
        'EMERGENCYSTATE_MODE': 'emergency_state_mode',
        'OBS_30_CNT_SOCIAL_CIRCLE': 'obs_30_cnt_social_circle',
        'DEF_30_CNT_SOCIAL_CIRCLE': 'def_30_cnt_social_circle',
        'OBS_60_CNT_SOCIAL_CIRCLE': 'obs_60_cnt_social_circle',
        'DEF_60_CNT_SOCIAL_CIRCLE': 'def_60_cnt_social_circle',
        'DAYS_LAST_PHONE_CHANGE': 'days_last_phone_change',
        'FLAG_DOCUMENT_2': 'flag_document_2',
        'FLAG_DOCUMENT_3': 'flag_document_3',
        'FLAG_DOCUMENT_4': 'flag_document_4',
        'FLAG_DOCUMENT_5': 'flag_document_5',
        'FLAG_DOCUMENT_6': 'flag_document_6',
        'FLAG_DOCUMENT_7': 'flag_document_7',
        'FLAG_DOCUMENT_8': 'flag_document_8',
        'FLAG_DOCUMENT_9': 'flag_document_9',
        'FLAG_DOCUMENT_10': 'flag_document_10',
        'FLAG_DOCUMENT_11': 'flag_document_11',
        'FLAG_DOCUMENT_12': 'flag_document_12',
        'FLAG_DOCUMENT_13': 'flag_document_13',
        'FLAG_DOCUMENT_14': 'flag_document_14',
        'FLAG_DOCUMENT_15': 'flag_document_15',
        'FLAG_DOCUMENT_16': 'flag_document_16',
        'FLAG_DOCUMENT_17': 'flag_document_17',
        'FLAG_DOCUMENT_18': 'flag_document_18',
        'FLAG_DOCUMENT_19': 'flag_document_19',
        'FLAG_DOCUMENT_20': 'flag_document_20',
        'FLAG_DOCUMENT_21': 'flag_document_21',
        'AMT_REQ_CREDIT_BUREAU_HOUR': 'amt_req_credit_bureau_hour',
        'AMT_REQ_CREDIT_BUREAU_DAY': 'amt_req_credit_bureau_day',
        'AMT_REQ_CREDIT_BUREAU_WEEK': 'amt_req_credit_bureau_week',
        'AMT_REQ_CREDIT_BUREAU_MON': 'amt_req_credit_bureau_mon',
        'AMT_REQ_CREDIT_BUREAU_QRT': 'amt_req_credit_bureau_qrt',
        'AMT_REQ_CREDIT_BUREAU_YEAR': 'amt_req_credit_bureau_year'
    }, inplace=True)
    return data


@pytest.fixture
def model(load_data):
    # Prepare data and train model
    data = load_data
    data.dropna(inplace=True)
    le = LabelEncoder()
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        data[col] = le.fit_transform(data[col])
    X = data.drop(['id', 'target'], axis=1, errors='ignore')
    y = data['target'] if 'target' in data.columns else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    numerical_columns = [
        'total_income', 'credit_amount', 'annuity_amount', 'goods_price'
    ]
    X_train[numerical_columns] = scaler.fit_transform(
        X_train[numerical_columns]
    )
    X_test[numerical_columns] = scaler.transform(
        X_test[numerical_columns]
    )
    rf = RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(
        X_train, y_train
    )
    rf.fit(X_resampled, y_resampled)
    return rf, X_test, y_test


def test_model_accuracy(model):
    rf, X_test, y_test = model
    y_pred = rf.predict(X_test)
    assert accuracy_score(y_test, y_pred) > 0.45, "Model accuracy is too low!"


def test_model_save(model):
    rf, _, _ = model
    joblib.dump(rf, 'loan_prediction_model.pkl')
    assert joblib.load(
        'loan_prediction_model.pkl'
    ) is not None, "Model was not saved correctly."
