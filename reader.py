import os
import pandas as pd


'''
split dataset
fraction is for the first half
'''
def splitByPortion(data, portion, rs):
    part1 = data.sample(frac=portion, random_state=rs)
    part2 = data.loc[~data.index.isin(part1.index), :]
    return (part1, part2)


data_dir = os.getcwd() + '/data/'


# encode boolean categorical column to numerical
def boolColumn(dfs, col, type1, type2):
    for df in dfs:
        for i in df.index:
            var = df.loc[i, col]
            if var == type1:
                df.loc[i, col] = -1
            elif var == type2:
                df.loc[i, col] = 1
            else:
                df.loc[i, col] = 0


# One-hot encoding with pre-defined missing value
def cateColumn(dfs, col, missing):
    types = dict()
    for df in dfs:
        for i in df.index:
            var = str(df.loc[i, col])
            if var != missing and var not in types:
                types[var] = col + '_' + var

    for val in types.values():
        for df in dfs:
            df.insert(0, val, 0)

    for df in dfs:
        for i in df.index:
            var = str(df.loc[i, col])
            if var in types:
                df.loc[i, types[var]] = 1
        df.drop(col, axis = 1, inplace = True)


# encode ordered categorical column to numerical
def numeColumn(dfs, col, values):
    for df in dfs:
        for i in df.index:
            var = df.loc[i, col]
            df.loc[i, col] = values[var]


# fill empty numerical column with mean value
def fillEmpty(dfs, col, missing, is_float):
    s = 0
    c = 0
    for df in dfs:
        for i in df.index:
            var = df.loc[i, col]
            if var != missing:
                s += float(var)
                c += 1
    if is_float:
        v = s / c
    else:
        v = int(s / c)
    for fd in dfs:
        for i in df.index:
            if df.loc[i, col] == missing:
                df.loc[i, col] = v


'''
(4177, 9)
(4177, 11)
'''
def abalone():
    data = pd.read_csv(data_dir + 'abalone/abalone.data', header = None)
    columns = ['sex', 'length', 'diameter', 'height', 'whole_weight',
               'shucked_weight', 'viscera_weight', 'shell_weight', 'class']
    data.columns = columns
    # cast categorical to numerical
    cateColumn([data], 'sex', '?')
    data.infer_objects()
    return data


'''
(226, 70)
(226, 95)
'''
def audiology():
    train = pd.read_csv(data_dir + 'audiology/audiology.data.csv', header = None)
    test = pd.read_csv(data_dir + 'audiology/audiology.test.csv', header = None)
    columns = ['age_gt_60', 'air', 'airBoneGap', 'ar_c', 'ar_u',
               'bone', 'boneAbnormal', 'bser', 'history_buzzing',
               'history_dizziness', 'history_fluctuating', 'history_fullness',
               'history_heredity', 'history_nausea', 'history_noise',
               'history_recruitment', 'history_ringing', 'history_roaring',
               'history_vomiting', 'late_wave_poor', 'm_at_2k', 'm_cond_lt_1k',
               'm_gt_1k', 'm_m_gt_2k', 'm_m_sn', 'm_m_sn_gt_1k', 'm_m_sn_gt_2k',
               'm_m_sn_gt_500', 'm_p_sn_gt_2k', 'm_s_gt_500', 'm_s_sn',
               'm_s_sn_gt_1k', 'm_s_sn_gt_2k', 'm_s_sn_gt_3k', 'm_s_sn_gt_4k',
               'm_sn_2_3k', 'm_sn_gt_1k', 'm_sn_gt_2k', 'm_sn_gt_3k',
               'm_sn_gt_4k', 'm_sn_gt_500', 'm_sn_gt_6k', 'm_sn_lt_1k',
               'm_sn_lt_2k', 'm_sn_lt_3k', 'middle_wave_poor', 'mod_gt_4k',
               'mod_mixed', 'mod_s_mixed', 'mod_s_sn_gt_500', 'mod_sn',
               'mod_sn_gt_1k', 'mod_sn_gt_2k', 'mod_sn_gt_3k', 'mod_sn_gt_4k',
               'mod_sn_gt_500', 'notch_4k', 'notch_at_4k', 'o_ar_c', 'o_ar_u',
               's_sn_gt_1k', 's_sn_gt_2k', 's_sn_gt_4k', 'speech',
               'static_normal', 'tymp', 'viith_nerve_signs', 'wave_V_delayed',
               'waveform_ItoV_prolonged', 'indentifier', 'class']
    bool_columns = ['age_gt_60', 'airBoneGap', 'boneAbnormal', 'history_buzzing',
                    'history_dizziness', 'history_fluctuating', 'history_fullness',
                    'history_heredity', 'history_nausea', 'history_noise',
                    'history_recruitment', 'history_ringing', 'history_roaring',
                    'history_vomiting', 'late_wave_poor', 'm_at_2k', 'm_cond_lt_1k',
                    'm_gt_1k', 'm_m_gt_2k', 'm_m_sn', 'm_m_sn_gt_1k', 'm_m_sn_gt_2k',
                    'm_m_sn_gt_500', 'm_p_sn_gt_2k', 'm_s_gt_500', 'm_s_sn',
                    'm_s_sn_gt_1k', 'm_s_sn_gt_2k', 'm_s_sn_gt_3k', 'm_s_sn_gt_4k',
                    'm_sn_2_3k', 'm_sn_gt_1k', 'm_sn_gt_2k', 'm_sn_gt_3k',
                    'm_sn_gt_4k', 'm_sn_gt_500', 'm_sn_gt_6k', 'm_sn_lt_1k',
                    'm_sn_lt_2k', 'm_sn_lt_3k', 'middle_wave_poor', 'mod_gt_4k',
                    'mod_mixed', 'mod_s_mixed', 'mod_s_sn_gt_500', 'mod_sn',
                    'mod_sn_gt_1k', 'mod_sn_gt_2k', 'mod_sn_gt_3k', 'mod_sn_gt_4k',
                    'mod_sn_gt_500', 'notch_4k', 'notch_at_4k', 's_sn_gt_1k',
                    's_sn_gt_2k', 's_sn_gt_4k', 'static_normal', 'viith_nerve_signs',
                    'wave_V_delayed', 'waveform_ItoV_prolonged']
    cate_columns = ['air', 'ar_c', 'ar_u', 'bone', 'bser',
                    'o_ar_c', 'o_ar_u', 'speech', 'tymp']
    data = pd.concat([train, test], ignore_index = True)
    data.columns = columns
    # drop id column
    data.drop('indentifier', axis = 1, inplace = True)
    # cast boolean to numerical
    for col in bool_columns:
        boolColumn([data], col, 'f', 't')
    # cast categorical to numerical
    for col in cate_columns:
        cateColumn([data], col, '?')
    data.infer_objects()
    return data


'''
(20867, 11)
(20867, 11)
'''
def avila():
    '''
    This part reads original txt file and save to csv
    train = pd.read_csv(data_dir + 'avila/avila-tr.txt', header = None)
    test = pd.read_csv(data_dir + 'avila/avila-ts.txt', header = None)
    columns = ['intercolumnar_distance', 'upper_margin', 'lower_margin',
               'exploitation', 'row_number', 'modular_ratio', 'interlinear_spacing',
               'weight', 'peak_number', 'modular_ratio', 'class']
    data = pd.concat([train, test], ignore_index = True)
    data.columns = columns
    data.to_csv('avila.csv', index=False)
    '''
    data = pd.read_csv(data_dir + 'avila/avila.csv')
    data.infer_objects()
    return data


'''
(286, 10)
(286, 16)
'''
def breast_cancer():
    data = pd.read_csv(data_dir + 'breast-cancer/breast-cancer.data.csv', header = None)
    columns = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps',
               'deg-malig', 'breast', 'breast-quad', 'irradiat']
    data.columns = columns
    # boolean columns
    boolColumn([data], 'node-caps', 'yes', 'no')
    boolColumn([data], 'breast', 'left', 'right')
    boolColumn([data], 'irradiat', 'yes', 'no')
    # categorical columns
    cateColumn([data], 'menopause', '?')
    cateColumn([data], 'breast-quad', '?')
    # numerical columns
    age_dict = {'?' : 0, '10-19' : 1, '20-29' : 2, '30-39' : 3, '40-49' : 4,
                '50-59' : 5, '60-69': 6, '70-79' : 7, '80-89' : 8, '90-99' : 9}
    numeColumn([data], 'age', age_dict)
    tumor_dict = {'?' : 0, '0-4' : 1, '5-9' : 2, '10-14' : 3, '15-19' : 4,
                  '20-24' : 5, '25-29' : 6, '30-34' : 7, '35-39' : 8,
                  '40-44' : 9, '45-49' : 10, '50-54' : 11, '55-59' : 12}
    numeColumn([data], 'tumor-size', tumor_dict)
    inv_dict = {'?' : 0, '0-2' : 1, '3-5' : 2, '6-8' : 3, '9-11' : 4, '12-14' : 5,
                '15-17' : 6, '18-20' : 7, '21-23' : 8, '24-26' : 9,
                '27-29' : 10, '30-32' : 11, '33-35' : 12, '36-39' : 13}
    numeColumn([data], 'inv-nodes', inv_dict)
    # move class to last column
    col_size = data.shape[1]
    data.insert(col_size - 1, 'Class', data.pop('Class'))
    data = data.infer_objects()
    return data


'''
(699, 10)
(699, 10)
'''
def breast_w():
    data = pd.read_csv(data_dir + 'breast-w/bc-w.data.csv', header = None)
    columns = ['id', 'Clump Thickness', 'Uniformity of Cell Size',
               'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
               'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
    data.columns = columns
    data.drop('id', axis = 1, inplace = True)
    data = data.infer_objects()
    # fill missing value with the average of column
    for col in data.columns:
        fillEmpty([data], col, '?', False)
    return data


'''
(1473, 10)
(1473, 22)
'''
def cmc():
    data = pd.read_csv(data_dir + 'cmc/cmc.data.csv', header=None)
    columns = ['W_age', 'W_education', 'H_education', 'Num_children', 'W_religion',
               'W_working', 'H_occupation', 'Standard-of-living',
               'Media', 'Contraceptive method used']
    cate_columns = ['W_education', 'H_education', 'H_occupation', 'Standard-of-living']
    data.columns = columns
    for col in cate_columns:
        cateColumn([data], col, '?')
    data = data.infer_objects()
    return data


'''
(30000, 24)
(30000, 31)
'''
def credit_card():
    '''
    This part reads original excel file and cast to csv
    data = pd.read_excel(data_dir + 'credit_card/credit_card.xls', header=1)
    data.drop('ID', axis = 1, inplace = True)
    data.rename(columns={'default payment next month': 'class'}, inplace=True)
    boolColumn([data], 'SEX', 1, 2)
    cate_columns = ['EDUCATION', 'MARRIAGE']
    for col in cate_columns:
        cateColumn([data], col, '0')
    data.to_csv('credit_card.csv', index=False)
    '''
    data = pd.read_csv(data_dir + 'credit_card/credit_card.csv')
    data.infer_objects()
    return data


'''
(366, 35)
(366, 35)
'''
def dematology():
    data = pd.read_csv(data_dir + 'dermatology/dermatology.data.csv', header=None)
    columns = ['erythema', 'scaling', 'definite borders', 'itching',
               'koebner phenomenon', 'polygonal papules', 'follicular papules',
               'oral mucosal involvement', 'knee and elbow involvement',
               'scalp involvement', 'family history', 'melanin incontinence',
               'eosinophils in the infiltrate', 'PNL infiltrate',
               'fibrosis of the papillary dermis', 'exocytosis', 'acanthosis',
               'hyperkeratosis', 'parakeratosis', 'clubbing of the rete ridges',
               'elongation of the rete ridges', 'thinning of the suprapapillary epidermis',
               'spongiform pustule', 'munro microabcess', 'focal hypergranulosis',
               'disappearance of the granular layer', 'vacuolisation and damage of basal layer',
               'spongiosis', 'saw-tooth appearance of retes', 'follicular horn plug',
               'perifollicular parakeratosis', 'inflammatory monoluclear inflitrate',
               'band-like infiltrate', 'Age', 'Class']
    data.columns = columns
    data = data.infer_objects()
    fillEmpty([data], 'Age', '?', True)
    return data


'''
(336, 8)
(336, 8)
'''
def ecoli():
    data = pd.read_csv(data_dir + 'ecoli/ecoli.data.csv', header=None)
    columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'Class']
    data.columns = columns
    data.drop('Sequence Name', axis = 1, inplace = True)
    data = data.infer_objects()
    return data

'''
(214, 10)
(214, 10)
'''
def glass():
    data = pd.read_csv(data_dir + 'glass/glass.data.csv', header=None)
    columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Class']
    data.columns = columns
    data.drop('Id', axis = 1, inplace = True)
    data = data.infer_objects()
    return data


'''
(270, 14)
(270, 21)
'''
def heart():
    data = pd.read_csv(data_dir + 'heart/heart.csv', header=None, sep=' ')
    columns = ['age', 'sex', 'chest_pain', 'resting_blood_pressure',
               'serum_cholestoral', 'fasting_blood_sugar',
               'resting_electrocardiographic', 'maximum_heart_rate',
               'exercise_induced_angina', 'oldpeak', 'slope',
               'major vessels', 'thal', 'class']
    bool_columns = ['sex', 'fasting_blood_sugar', 'exercise_induced_angina']
    cate_columns = ['chest_pain', 'resting_electrocardiographic', 'thal']
    data.columns = columns
    # cast boolean to numerical
    for col in bool_columns:
        boolColumn([data], col, 0.0, 1.0)
    # cast categorical to numerical
    for col in cate_columns:
        cateColumn([data], col, '?')
    data.infer_objects()
    return data


'''
(155, 20)
(155, 20)
'''
def hepatitis():
    data = pd.read_csv(data_dir + 'hepatitis/hepatitis.data.csv', header=None)
    columns = ['Class', 'AGE', 'SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE',
               'MALAISE', 'ANOREXIA', 'LIVER BIG', 'LIVER FIRM',
               'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES', 'BILIRUBIN',
               'ALK PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME', 'HISTOLOGY']
    bool_columns = ['STEROID', 'FATIGUE', 'MALAISE', 'ANOREXIA',
                    'LIVER BIG', 'LIVER FIRM', 'SPLEEN PALPABLE',
                    'SPIDERS', 'ASCITES', 'VARICES']
    miss_columns = ['BILIRUBIN', 'ALK PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME']
    data.columns = columns
    for col in bool_columns:
        boolColumn([data], col, '1', '2')
    data = data.infer_objects()
    for col in miss_columns:
        fillEmpty([data], col, '?', True)
    # move class to last column
    col_size = data.shape[1]
    data.insert(col_size - 1, 'Class', data.pop('Class'))
    return data


'''
(10299, 562)
(10299, 562)
'''
def human_activity():
    train = pd.read_csv('data/human-activity/train.csv')
    test = pd.read_csv('data/human-activity/test.csv')
    data = pd.concat([train, test], ignore_index = True)
    data.drop('subject', axis = 1, inplace = True)
    return data


'''
(300, 5)
(300, 5)
'''
def iris():
    iris = pd.read_csv(data_dir + 'iris/iris.csv', header=None)
    bezd = pd.read_csv(data_dir + 'iris/bezdekIris.csv', header=None)
    data = pd.concat([iris, bezd], ignore_index = True)
    return data


'''
(148, 19)
(148, 39)
'''
def lymphography():
    data = pd.read_csv(data_dir + 'lymphography/lymphography.data.csv', header=None)
    columns = ['class', 'lymphatics', 'block of affere', 'bl. of lymph. c',
               'bl. of lymph. s', 'by pass', 'extravasates', 'regeneration of',
               'early uptake in', 'lym.nodes dimin', 'lym.nodes enlar',
               'changes in lym.', 'defect in node', 'changes in node',
               'changes in stru', 'special forms', 'dislocation of',
               'exclusion of no', 'no. of nodes in']
    cate_columns = ['lymphatics', 'changes in lym.', 'defect in node',
                    'changes in node', 'changes in stru', 'special forms']
    data.columns = columns
    for col in cate_columns:
        cateColumn([data], col, '?')
    data = data.infer_objects()
    # move class to last column
    col_size = data.shape[1]
    data.insert(col_size - 1, 'class', data.pop('class'))
    return data


def read(dataset):
    data_map = {
        'abalone': abalone,
        'audio': audiology,
        'avila': avila,
        'bc': breast_cancer,
        'bw': breast_w,
        'cmc': cmc,
        'credit': credit_card,
        'demat': dematology,
        'ecoli': ecoli,
        'glass': glass,
        'heart': heart,
        'hepa': hepatitis,
        'ha': human_activity,
        'iris': iris,
        'lymph': lymphography
    }
    return data_map[dataset]()


if __name__ == '__main__':
    data = avila()
    print(data.shape)

