import numpy as np


def f(X):
    import pandas as pd
    from sklearn.preprocessing import PolynomialFeatures

    poly_degree = 8

    def standarize_matrix(data_matrix):
        mean = np.mean(data_matrix, axis=0)
        std = np.std(data_matrix, axis=0)
        std[std == 0] = 1
        standarized = (data_matrix - mean) / std
        return standarized

    def standarize_matrix_and_add_ones(data_matrix):
        standarized = standarize_matrix(data_matrix)
        return np.c_[np.ones(standarized.shape[0]), standarized]

    def preprocess_data(data_df):
        nonlocal poly_degree
        columns = pd.DataFrame()
        columns[0] = data_df[0]

        columns[1] = data_df[1]

        columns[2] = data_df[2]
        columns[3] = data_df[3]
        columns[4] = data_df[4]

        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        columns_matrix = poly.fit_transform(columns)

        columns_matrix = standarize_matrix_and_add_ones(columns_matrix)
        return columns_matrix

    X_df = pd.DataFrame(np.copy(X))
    X_matrix = preprocess_data(X_df)

    coefs_string = None
    coefs_string = """ 
        [ 9.72433012e+02 -2.80655661e+03  5.12824191e+02  9.70285021e+03
    1.03977258e+04  1.87723283e+04 -5.85201686e+03  4.70619262e+03
    5.21818247e+03 -2.32432258e+03  1.08812082e+04 -4.59494407e+03
    1.41065873e+03  3.49155514e+03 -5.15425567e+03 -6.31551280e+03
    -1.17190520e+04 -5.11650415e+04 -1.09610595e+04 -4.60029892e+04
    -6.12213184e+04  4.63599179e+03 -1.97121734e+03  1.10965407e+04
    2.93986234e+02  2.32870329e+04 -1.34544890e+03 -8.32526759e+03
    -1.63744559e+03 -1.87448675e+04  4.94637768e+03 -1.13399298e+03
    -2.62820634e+04 -9.38359544e+03  1.79118031e+04 -1.26931337e+04
    1.08443258e+03  1.24047051e+04 -3.07447021e+03  1.71249541e+04
    -7.75402091e+03 -1.45306213e+04  5.74538303e+03  5.16032147e+03
    -8.59407710e+03  1.11169281e+04 -2.24144841e+03  9.76497161e+03
    2.89168642e+04  1.76024033e+04  3.03484931e+04  1.07077488e+05
    3.05651030e+04  8.44466932e+03  8.43293372e+04  5.00250742e+04
    1.56270841e+03 -6.30657378e+02 -1.84500532e+03  6.57275102e+02
    -2.48636998e+04 -7.04391854e+02 -4.30682772e+03 -1.96737296e+03
    1.44513688e+04 -7.86707327e+02 -6.92826236e+03 -5.19098227e+04
    2.91097996e+03  3.86678592e+03 -3.57579611e+04  1.39962489e+03
    1.69323198e+03  1.84733955e+03  3.38548669e+03 -2.18933373e+03
    4.03848620e+03  4.17047406e+04  2.68942708e+02  3.48138386e+03
    2.53505334e+04 -7.49133733e+03  8.12001358e+03  5.51914444e+03
    -9.81924356e+03 -4.46179793e+03  2.43540167e+04  1.24905533e+04
    3.71896501e+04 -4.50613221e+04  6.71265492e+03  3.79526627e+02
    3.22074680e+03 -2.12957381e+02 -7.90032555e+03 -2.52473246e+03
    -3.33614453e+02 -5.46827105e+04  4.65096904e+03  1.32605621e+04
    -2.17922255e+04 -1.57730592e+03  2.52028163e+04  2.47802862e+04
    7.35933685e+03  3.69303229e+04 -2.95309994e+04 -5.69914556e+03
    -2.16198431e+04  1.30750543e+03 -3.50505959e+03 -6.67269738e+03
    -5.00989892e+03  8.88180078e+03 -1.87028173e+04 -6.87387514e+03
    -2.44471313e+04 -4.09639089e+04 -1.26454490e+04 -2.41106610e+04
    -8.58549966e+04  4.70834849e+04 -7.29296308e+04  2.50949945e+04
    -5.80051161e+04  3.22982773e+04  7.88631622e+02  5.51335370e+01
    3.43715311e+02 -4.84288255e+02 -6.49843297e+03  1.24236817e+02
    2.73547275e+02  7.60410640e+02  1.49511826e+03  3.71517893e+03
    1.38705225e+02  4.48672768e+03 -1.31277368e+02 -2.71614862e+03
    5.58987363e+04  4.02286391e+02 -5.98575312e+02  1.09210115e+00
    3.57105258e+03  2.44954701e+03 -5.28559488e+02  1.55625590e+04
    9.54192658e+02  7.41603892e+03 -4.01149930e+04 -1.16010066e+04
    9.87340293e+03  1.75567731e+04  1.10932578e+03  1.19942062e+04
    1.00337267e+05 -5.54935026e+03 -5.35388950e+03 -1.26299445e+04
    1.69046074e+04  1.11336708e+02  2.09866061e+02 -1.74113839e+02
    -5.93135132e+03 -3.62720407e+02  8.32666792e+02 -5.81219145e+03
    1.97593425e+02 -8.25370153e+03 -7.68712055e+02  1.92645546e+03
    -2.38203829e+03  1.00940592e+04  3.79446046e+03 -1.66290800e+04
    -9.31289343e+04  6.37279310e+01 -4.48269125e+03  4.57369097e+03
    -1.78477487e+02 -1.53711371e+04 -5.30802523e+02  6.26618448e+03
    1.91613962e+04 -4.35274813e+04  1.24908188e+04  7.16664548e+03
    3.93045414e+03  4.92934549e+04  3.21369880e+03 -3.91871921e+03
    -5.21987197e+04 -3.82683928e+04  3.43623252e+04 -6.38480366e+03
    1.20566593e+02 -1.69709668e+01 -2.14272430e+02 -1.75668838e+03
    -6.42086289e+03 -3.17157578e+03 -4.80028486e+03 -1.53235687e+03
    4.24948309e+03  1.96193580e+04 -5.62592666e+03  1.90181826e+03
    1.39783500e+04 -6.30779192e+03  6.26484182e+03  9.98520490e+04
    -1.26718129e+03 -1.36821465e+04 -2.79852072e+04  3.55118787e+03
    6.20084456e+03 -8.15448599e+03 -3.49952346e+03 -1.45279172e+03
    -9.05656538e+04 -1.36174910e+04 -1.55212937e+03 -2.70874912e+04
    -1.05003899e+02  2.06970025e+04  1.80547023e+03  1.75928047e+04
    4.18862508e+04  6.97056132e+03 -6.96797614e+03  1.96365608e+04
    -3.86189949e+03 -1.17017164e+04  8.56284097e+03  1.00255881e+04
    -2.13280215e+04  4.85549896e+04 -1.62427968e+04 -2.36818612e+04
    -3.17958533e+04  2.22311685e+04  5.97315438e+04  2.32105651e+03
    -3.47108488e+04 -1.88483083e+04 -2.54014514e+05  7.20267132e+04
    6.44805992e+04 -6.26393633e+04 -2.06205651e+04 -5.17727289e+04
    1.26475792e+02  1.88612317e+01 -3.89451895e+02 -3.73803856e+01
    -2.28275702e+03  8.73261923e+00 -1.38599549e+02 -7.57934762e+01
    -2.45635540e+02  6.82186305e+02  4.73192269e+02 -2.31390887e+03
    -1.99726625e+02  1.12036586e+03  1.14531517e+04  1.04511774e+02
    -3.17262768e+02  2.87405839e+01 -4.45203384e+01 -4.33314866e+02
    -6.59981932e+02  4.91549330e+02  1.82603967e+02 -1.95377013e+03
    -1.66975751e+03 -2.72437810e+03 -7.90999975e+02 -6.88670756e+03
    -2.21909905e+02  9.76914318e+02 -4.30970703e+03  4.59868599e+02
    -2.03390419e+02  4.07543259e+03 -6.44867623e+04  4.54581618e+01
    -5.78905343e+01  3.38444974e+01 -1.29843547e+03  1.14250797e+03
    6.33430289e+02 -1.44130469e+02 -2.97595717e+02 -2.50076154e+02
    -6.04787909e+03 -1.03052322e+03  5.33688463e+02 -6.44900482e+03
    3.70710698e+02 -2.29034709e+02 -2.07337773e+04 -1.58427156e+02
    -2.74928006e+03 -9.95353597e+03  5.34520024e+04  1.44256011e+04
    -7.78939221e+03  5.07208892e+03 -3.12661533e+02 -1.53514950e+04
    -2.91066386e+04  6.43449606e+02 -3.40892467e+03 -3.65274259e+03
    -1.01931881e+05  6.34957074e+03  2.49969761e+03  8.31957129e+03
    1.32165600e+04  1.48552641e+04  2.16073203e+01 -1.55734159e+02
    -1.18862153e+02 -9.25199586e+01 -3.93606870e+02 -4.95381257e+02
    3.11328042e+02 -3.51706260e+01  1.04404390e+03  9.11942930e+03
    -1.22802871e+03 -2.37751345e+03  4.31669345e+03 -1.00295113e+03
    2.09519895e+03  4.67781971e+03 -1.17866604e+02  4.89556770e+02
    1.18166758e+04 -4.37159061e+03  1.14739907e+03  2.74607062e+03
    -1.18769752e+04 -4.24290084e+03  6.99326247e+03 -1.00806704e+04
    -8.62986821e+02 -3.92250030e+03  2.31503383e+04  1.08371478e+05
    -9.03859694e+02  2.65255410e+03  6.64530530e+03 -1.71201040e+04
    -3.35970023e+04  3.66997297e+04 -5.08674975e+03  1.51205567e+04
    -1.20118285e+04  1.90202599e+04 -4.01532190e+04 -1.15703925e+04
    -1.83713399e+04  6.23425067e+04 -1.80087788e+04 -7.87800005e+02
    -1.62883059e+03  6.88900003e+03 -1.02000767e+05 -1.64047160e+04
    1.43102552e+03  9.83222063e+03  7.74807294e+04 -6.18149112e+03
    2.49772681e+04  8.01733216e+03 -6.74948357e+01  5.24954774e+02
    8.32624938e+01 -1.26333447e+03 -1.11293141e+02  4.83544022e+01
    9.23294076e+02  5.83151487e+02 -2.14289195e+02  2.66811156e+03
    4.04559444e+03  3.00835990e+03  1.09896626e+04  1.60726443e+03
    3.80461817e+03  5.74159890e+02  1.50810181e+03  1.34090990e+03
    -8.77774693e+03 -2.31783362e+04 -4.41991575e+02 -7.16351533e+03
    2.90640475e+04  6.31049529e+03  1.40677684e+03 -5.09191583e+04
    -2.10100157e+02  1.04303611e+04 -1.83285747e+04 -7.52781591e+04
    1.41659786e+03  2.12798971e+03  1.74702731e+04  3.50148639e+04
    1.04157482e+04 -1.80536631e+03 -8.24641262e+03 -2.21382624e+03
    -4.42752358e+03  5.33550854e+04 -1.91812569e+04 -3.02020051e+03
    2.07926307e+04  8.11971626e+04  8.26589232e+03 -3.02206435e+03
    1.49922801e+04  1.60582822e+04 -5.04201008e+04  1.96807427e+04
    2.02950699e+03 -6.30549138e+03 -2.59093661e+04 -3.52465855e+04
    -2.34533409e+03 -6.14505312e+02 -9.60199331e+03  2.01759311e+03
    -2.24801437e+03  7.90823282e+03 -3.03744155e+03  7.04695222e+03
    -4.35219660e+04  2.92234117e+04  1.92071415e+03  3.30898674e+04
    -2.10937299e+04 -1.89636073e+04  7.34162829e+03  5.35075247e+04
    7.73074362e+04 -1.49936578e+03 -3.67738310e+04 -4.42142535e+04
    1.26574468e+04  8.37243072e+04  6.85293722e+04  3.66036622e+05
    -7.17008436e+03 -9.83834484e+04  1.09603005e+04  3.95769563e+04
    5.47657141e+04 -1.07406455e+04  3.74044288e+01  2.90803395e+01
    -4.25514093e+01 -8.19808066e+00 -2.78273062e+02 -4.04070846e-01
    -9.38784302e-01 -2.08743509e+01  1.25486175e+01  4.16996952e+02
    9.78843100e+01  1.67544756e+02 -2.67732315e+01  1.07411226e+02
    2.40478214e+03  2.74150995e+01 -1.91168321e+01  4.47483690e+00
    4.27953717e+01  1.41439462e+02  1.02816190e+02  7.84707714e+01
    8.21497704e+01 -2.40042481e+01  3.76437155e+02 -1.24080162e+03
    1.79982899e+01  1.28933157e+03 -1.04248673e+02 -8.41319199e+02
    1.86735311e+03  2.45215841e+02  2.59691074e+02 -1.02716221e+03
    -8.95906361e+03 -1.85170268e+01 -1.87578799e+01 -2.15870574e+01
    -1.79013810e+02  1.06024605e+02  1.26411912e+02  3.53729427e+02
    8.13136149e+01 -2.35717209e+02 -6.26971878e+01  1.51379368e+02
    3.80423974e+02  1.48533281e+02  1.14872307e+02  6.20018685e+02
    -9.51224801e+02 -1.12724584e+02 -2.11680623e+02  1.94685429e+03
    9.23405459e+02  6.11806378e+02 -2.73678935e+02  4.61228315e+03
    2.55123830e+02  1.66136908e+03  2.54110987e+03  1.25439887e+02
    -7.17370458e+01 -2.30420984e+03  3.25436467e+03 -2.21283088e+02
    -5.29362122e+02  7.65089741e+02 -2.69928902e+03  3.72956528e+04
    3.93268593e+01 -7.38985292e+00  1.79103957e+01 -9.85356127e+01
    5.29284610e+01  6.63931504e+01 -1.50717702e+01 -1.57897460e+02
    8.26918161e+01  1.37914643e+03 -1.16420298e+03 -1.93983962e+02
    1.02880693e+02 -1.15338747e+02 -8.46540569e+02  4.50922952e+02
    1.64599798e+02  4.42851113e+02  3.77708864e+02  4.55485887e+03
    5.18774872e+02 -8.20372962e+01  4.37896967e+02 -3.99550629e+02
    -1.53525096e+01  7.00052919e+03  4.33701190e+02 -6.20972629e+02
    4.73589901e+02  1.17129381e+04 -2.81955625e+02  5.07685786e+02
    2.55757423e+03  6.09550375e+03 -3.46119957e+04 -6.34635935e+03
    4.07338905e+03 -1.20735385e+04 -1.15226894e+03  6.51464872e+03
    1.00891894e+04  2.52803862e+02  3.03761495e+03  8.25048574e+03
    1.15701732e+04 -4.71519879e+02 -9.82018073e+02  1.87843959e+03
    -4.02982032e+03  5.74396244e+04 -3.60239673e+03 -1.77458480e+03
    -3.00051912e+02 -6.78144501e+03 -4.63117068e+03 -2.06580279e+04
    -1.68161896e+01  3.19110089e+01 -1.55146286e+01  3.97866440e+01
    3.17681602e+01  8.46029979e+01  1.31757112e+02  2.32595456e+01
    8.46265273e+01  1.30330182e+02  1.69947674e+01  2.46758905e+02
    5.17794816e+02  1.69778683e+02  3.89472420e+02 -8.96626091e+02
    3.78767877e+01 -2.46130374e+02 -1.10520483e+03 -6.22924452e+03
    1.96935833e+03  1.19873698e+03 -3.35014786e+03  7.16428699e+02
    1.53877552e+03 -9.60973058e+02  1.16675763e+02  5.11010312e+02
    -3.82701517e+03 -2.00302747e+03  3.12162327e+01  2.47844870e+01
    -7.65460394e+02 -7.02342122e+03  4.71970414e+03 -2.12360630e+03
    -3.76315843e+03  7.67202144e+03  2.75119287e+03  1.18874065e+03
    2.41967898e+03  3.12185846e+02  2.01651146e+03 -1.02218174e+04
    1.02334029e+04 -1.18392596e+02  1.29768414e+03  1.64299828e+03
    -1.22527600e+04 -6.74350163e+04  6.95151786e+02  3.72563791e+02
    -4.33766243e+03 -2.45867063e+03  1.46851739e+04  3.30024983e+04
    -3.42347823e+04  7.81977068e+03 -3.40391786e+03  4.81293918e+03
    -1.89438245e+04  5.44136242e+03  6.24503349e+03  3.27320958e+03
    8.33731288e+03  1.32852342e+04  9.98806824e+02  7.40362312e+03
    1.07561304e+04 -6.28777525e+04  3.27941675e+04 -1.35589726e+03
    3.49284244e+03 -8.95304822e+03 -4.09962776e+03  9.46301324e+04
    -7.45441041e+03  3.70745436e+02 -3.09812444e+03 -8.36406651e+03
    -4.99961200e+04  2.84593194e+04 -5.08270194e+04 -1.11491665e+02
    7.90904430e+01 -5.06529140e+01  5.85603548e+01  2.14258374e+02
    -8.67245045e+01 -1.01721521e+02 -9.15772412e+02  6.43705534e+00
    -1.93938718e+01  2.13668065e+03  1.45798333e+01 -3.68422980e+02
    3.17293788e+02 -1.00492083e+02  4.05928630e+02 -1.91302204e+03
    -1.35050938e+02 -7.90603643e+02  7.05511879e+02 -1.52947148e+03
    -5.27516500e+02 -2.02803030e+03 -5.66210176e+03 -2.86049304e+02
    -2.11100430e+03 -6.03756373e+03 -8.61598128e+02 -1.38156654e+03
    -1.17552579e+03  2.17448621e+03 -3.36735534e+02 -1.45784961e+03
    4.85619117e+00  6.63808940e+03  1.33485540e+04  5.10545088e+03
    6.13526657e+03 -2.22869380e+04 -2.30113363e+03  5.77719897e+02
    -3.24430262e+03 -1.33629218e+02 -7.40456412e+03  2.30999131e+03
    4.03215238e+04  7.48724776e+02 -8.81275798e+02 -4.99755145e+03
    1.55823572e+04  1.96776160e+04 -1.36303297e+03 -4.99032394e+01
    -2.47670025e+03 -1.07628084e+04 -2.36546174e+04 -4.59842970e+03
    -5.81851590e+03  1.27041773e+04  1.25051017e+04  2.01459156e+02
    -2.61088112e+04 -5.94597419e+03  4.62114752e+03 -5.17993822e+01
    -2.07826742e+04  2.89621245e+04  1.54515183e+03 -7.19636614e+03
    -1.73436087e+04 -3.89063601e+04 -1.45726428e+04  1.68561842e+03
    -3.99452904e+02 -9.54040734e+03  2.95034977e+03  4.60437114e+04
    -2.85378494e+04 -1.31866826e+03 -2.23801690e+03  9.76711415e+03
    1.34881113e+04  1.28404056e+04 -3.84034617e+03  9.19456112e+03
    -6.73645328e+03  6.26610018e+03  7.60515650e+03 -1.00326679e+04
    -1.20033645e+04  6.21577759e+03  2.57218315e+04 -3.21607434e+04
    2.53286360e+04 -9.68752637e+03  1.19548900e+04 -5.76049253e+03
    2.23663324e+04 -4.49892384e+04 -1.99158791e+04  3.06235106e+03
    1.22015134e+04  1.13752203e+04 -2.75859417e+04 -1.27390283e+04
    -5.62920898e+04  2.14075438e+02 -1.39934698e+03  3.23107785e+04
    4.02825057e+03  2.24570789e+03 -7.33262174e+04 -3.31842134e+04
    -2.37423333e+05 -1.20027956e+03  1.13213518e+04  5.87708453e+04
    -4.36392495e+04  8.91793904e+02 -2.71487827e+04  3.36643736e+04
    1.31271865e+01 -5.37094422e+00 -5.90591382e+00 -1.66674585e+01
    -1.59063823e+01 -5.64292960e+00 -1.92509352e+01 -1.18321511e+00
    -4.48031271e+00  7.63538750e+00  3.82339300e+00  1.77916138e+01
    -4.06515087e+00  1.05920547e+01  1.47116155e+02 -4.62075037e+00
    -1.20148324e+00 -4.36788269e+00 -1.53041363e+00 -2.34587204e+01
    3.41569278e+00  2.64891783e+01  9.26627224e+00  4.97923687e+00
    -2.09203264e+01 -1.67096372e+02  5.90533058e+00 -9.60259125e+01
    -9.75950349e+00 -8.93826565e+01  2.28432671e+01  6.35988931e-01
    2.44243738e+01 -3.89160729e+01 -8.91435864e+02 -7.34457668e+00
    -7.65104400e-01 -8.98513325e-01 -3.20366971e+01  1.97662062e+00
    3.67473292e+00  5.28001958e+00 -8.67401125e+00  4.77009421e+00
    -3.47910636e+01 -3.06844533e+01 -3.80703523e+01 -5.38325939e+01
    -2.22184013e+01 -1.15490192e+01 -7.25957247e+00 -7.49730367e+00
    -5.14901707e+01  4.76312761e+01 -1.83459300e+02  4.92494818e+02
    2.21158574e+01 -1.89234414e+02 -4.03933018e+01 -1.86478650e+01
    -5.03222275e+02 -9.36707315e-01  1.73517475e+02  3.39019633e+02
    -5.31376833e+02 -8.05701519e+01 -8.34645344e+01 -1.67929446e+02
    3.76756126e+02  2.58913234e+03  1.20541123e+00  2.10147681e-01
    6.27176862e+00  7.37903778e+00  2.29932793e+01 -1.59261944e+01
    -4.14074295e+00  5.67034333e+00  3.44242742e+01  7.88873952e+01
    6.99018872e+00 -7.64398974e+01 -5.17738718e+01 -4.29588318e+00
    -8.35567633e+00 -1.35268215e+02 -4.38346079e+01 -6.40174517e+00
    1.33174284e+02  3.63946395e+01  1.20555991e+02 -9.51704346e+01
    -4.06520799e+02 -3.05846118e+01 -1.91116504e+02  5.37850724e+02
    2.01116982e+01 -9.60029178e+01 -6.34161869e+01  1.86031078e+01
    -2.80845312e+01  1.43891378e+02 -3.21040560e+00 -7.22386905e+02
    -7.44802235e+01 -4.27501029e+01  4.63889556e+02 -9.35553237e+02
    -1.14698482e+02 -8.38839149e+02 -6.92386794e+02 -6.31019189e+00
    9.92941332e+00  8.72344293e+01 -7.82181723e+02 -5.18677291e+01
    -1.21629454e+01  2.74577357e+01  8.36225419e+02 -8.54463475e+02
    -8.50545770e-01  2.58706767e+02  2.87888924e+01 -3.80843025e+02
    7.29904696e+02 -8.62510469e+03  1.26073745e+01 -7.34806511e+00
    1.95062426e+00 -2.02122247e+01  1.07816805e+01  9.63466564e+00
    -2.10450644e+01 -1.94158642e+00 -2.40479184e+01  6.95244674e+01
    -3.66436465e+00 -2.87291194e+01 -7.10531882e+00 -6.03368251e+00
    -1.01847016e+01  1.71870435e+01  1.79140121e+01  1.38780970e+02
    -1.48666018e+02 -4.58959062e+02  3.58346409e+02  1.29992712e+01
    1.92916178e+02  4.45949807e+01  9.76014790e+01 -2.76945831e+02
    -3.15079008e+00  5.19812303e+01  3.38373196e+02 -1.40971638e+02
    -4.73711866e+01 -6.59974167e+01 -1.77862023e+02 -1.73113460e+02
    -1.29715711e+03 -1.22673705e+02 -2.09186460e+02  3.14752761e+02
    6.49282137e+01  4.78719933e+02 -1.00340578e+03  2.54526851e+00
    2.19524553e+02 -6.56038946e+02 -1.73492680e+03 -8.33323085e+01
    -2.62198737e+02  3.61056625e+02  1.24653857e+02 -2.66526687e+03
    1.93169221e+01  2.89652920e+02 -4.42239409e+02 -7.67308852e+02
    -1.50058138e+03  8.84787166e+03  7.54359566e+02 -6.49077881e+02
    3.88577648e+03  1.81540031e+02 -2.16965725e+03  3.57480853e+02
    2.86690778e+02 -6.47174743e+00 -4.38185036e+02 -4.37301770e+03
    -9.92276487e+01 -5.50758925e+02 -1.19087916e+03 -2.31854637e+03
    2.30636191e+02 -1.26448154e+01  6.62850844e+02  2.86307455e+02
    -2.87147282e+02  2.65886883e+03 -1.45900992e+04  8.56575795e+02
    4.53590511e+02  5.24495836e+01 -5.85759597e+01  2.18184870e+03
    -1.47923035e+02  7.02410421e+03 -3.53319794e+01  2.04080494e+01
    1.35520563e+01 -3.76274313e+01 -1.38857614e+01  1.97270858e+01
    -2.07083299e+01  8.79828433e+00 -4.16637939e+00 -2.17299853e+01
    -1.31901348e+01 -1.94686292e+01  4.05968538e+00  4.31850064e+01
    -1.07314149e+02 -1.77653937e+01  1.32964318e+01 -7.25972584e+01
    4.73406315e+01 -8.57252038e+01  7.28938813e+01 -9.06141987e+01
    -1.27789285e+02  3.25723437e+01 -8.88463772e+01 -1.08932147e+02
    -2.14293819e+01 -1.53569391e+02 -4.35753076e+01  4.29587791e+02
    -4.68660652e+00  2.14677240e+01  1.87481125e+02  3.16223737e+02
    1.60678488e+03 -7.47629151e+02 -2.65403054e+02  5.97940288e+02
    -1.59392414e+02 -2.13617570e+02  1.03019553e+03 -6.67596460e+01
    -2.30804590e+02 -5.04868871e+02 -5.11627345e+02  6.66261769e+01
    -1.22215339e+02  3.41088736e+01  1.77190060e+03  5.37109207e+02
    -1.72198191e+01 -5.51529420e+01  1.47351425e+02  1.84966037e+02
    1.39061695e+03 -1.48080675e+03  8.57723984e+02  1.30188613e+03
    -2.45828259e+03 -6.28648919e+02  8.44186403e+01  1.90842841e+02
    4.57199577e+00 -9.73152074e+02 -1.45609277e+01 -1.20146009e+03
    -1.21052359e+02 -3.31684125e+01  1.29294272e+02  3.70491509e+03
    -3.52746877e+03  1.86243982e+02 -1.42031600e+02 -5.20230068e+02
    -3.82434251e+02  2.08408360e+03  1.75881315e+04 -2.30765986e+02
    -1.70144878e+02  7.90624228e+01  1.87816722e+03 -2.68354039e+02
    -4.00447199e+03 -1.03963856e+04  1.29864280e+04 -1.60428823e+03
    -9.36422668e+03 -1.91136238e+03  3.80349946e+02  2.01396121e+04
    -3.58868825e+02  1.87367683e+03  7.66359392e+03 -2.93406874e+04
    -1.18345989e+03 -2.78111107e+03 -2.01830285e+03 -9.96550842e+03
    2.31394836e+04  1.01570743e+03 -1.25804401e+03  8.48165118e+02
    -3.53756484e+03  2.66679211e+04 -2.82524493e+04  1.87584525e+01
    -1.29979018e+02 -5.82779268e+02  3.50461773e+03  8.06516530e+02
    -3.39294605e+04  1.37989943e+04 -2.07782016e+02  3.64786406e+02
    1.22755749e+03  2.09278700e+03  1.24799614e+04 -1.16652843e+04
    2.07721770e+04 -3.10138594e+03 -4.07422031e+00 -3.43975471e+00
    -3.93205445e+00 -9.46875776e+01  5.64964098e+01  3.38694654e+01
    -7.16789416e+01 -9.14314383e+00 -7.44316274e+01 -1.02233578e+02
    -7.09783135e+01  8.15692968e+01  2.08988600e+02 -1.05810045e+01
    3.11524853e+00  3.41280066e+02 -1.85860628e+01  1.81633768e+01
    -2.61775442e+01 -9.89423135e+02 -1.18590670e+02  1.70482301e+02
    1.28733952e+02 -6.13555653e+01  1.12752253e+02 -2.86388189e+02
    -4.43032637e+01  2.55008621e+02 -4.54223281e+02  1.17519958e+03
    1.87493788e+01  1.05699361e+02  2.13658573e+02 -2.56590143e+02
    1.52166394e+02 -1.03653988e+02  5.54744594e+02  4.25988639e+02
    -2.78782041e+01  6.76865979e+02  2.42768182e+03  9.60385725e+01
    1.85492203e+02  2.82391745e+02  4.55916676e+02  2.46404583e+02
    9.63117034e+01  5.24948392e+02 -1.26358355e+02 -6.59938422e+02
    -9.49309715e+01  3.65682330e+02  3.51851750e+02 -3.13832328e+02
    -1.65762705e+03 -3.07083892e+03 -2.56598583e+03 -1.62443062e+03
    6.70480174e+03  3.13850591e+02 -1.02246328e+03  9.35397242e+02
    6.74398165e+01  1.17138238e+03  5.44316954e+02  5.51396151e+02
    -1.64375395e+02  3.47123593e+02  2.42932789e+03 -2.43731290e+03
    -1.16507327e+04 -1.12380221e+02 -1.99952644e+02  4.31058781e+02
    2.24807011e+02 -3.74561054e+03  9.72215283e+02  1.45800379e+02
    1.04383984e+03 -1.34001387e+03  1.92942822e+03  2.36134699e+03
    6.43130783e+03 -4.75189889e+02  3.39159256e+03 -5.15979325e+03
    -4.52165265e+03  9.05740963e+02  7.20895656e+03 -1.25824938e+03
    -1.31614527e+03 -9.00539337e+02  1.51944793e+03  3.71297448e+03
    -9.24352812e+02  1.15503725e+03  4.33058145e+02  5.95966940e+03
    -1.36938996e+04 -1.89696021e+02  8.50302957e+02  1.63294834e+03
    5.40451844e+03  6.11762167e+03  9.57357285e+03 -3.48284972e+02
    -2.47354636e+02 -2.78102114e+02  2.84380511e+03 -4.52813645e+03
    -1.22371484e+04  8.25206327e+03  3.27624614e+02  4.76433810e+02
    8.41315960e+02 -4.74774582e+03 -1.15553710e+03 -1.63712621e+03
    2.04757751e+03 -4.24943403e+03  5.96698649e+03 -6.56617790e+03
    -4.95068082e+03  5.33172155e+03  1.75609046e+04 -5.43855152e+03
    -8.83048992e+03  4.83613075e+03 -3.17866422e+04  1.54376447e+04
    -2.98076448e+03  1.09398413e+04 -5.45785833e+02  3.31095847e+04
    -2.32045095e+04 -3.51471507e+02 -3.24864117e+03 -9.27406979e+03
    -3.65812930e+03 -8.83377661e+03  2.68969242e+04 -1.65458772e+03
    2.74481289e+03 -6.57331174e+03  7.93094906e+03  7.12753941e+03
    7.71383797e+03  3.61558324e+03  3.54875011e+01  2.08021237e+03
    -5.11736992e+03 -3.69554672e+03 -2.05871123e+03 -2.19820003e+03
    2.05059762e+04  6.45272291e+03  5.83591726e+04  1.52752889e+03
    -3.74789937e+03  1.36905173e+03 -1.80881089e+04  1.99296754e+04
    -6.87748473e+03  3.99002059e+03 -1.13693776e+04]
    """

    # with open("parameters", "r") as f:
    #     coefs_string = f.read()

    coefs_string = coefs_string.replace("[", "").replace("]", "")
    coefs = np.fromstring(coefs_string, sep=" ")
    # print(coefs.shape)

    # with open("loaded_parameters", "w") as f:
    #     f.write(np.array2string(coefs))

    prediction = X_matrix @ coefs

    return prediction


data = np.loadtxt("dane.data")

X = data[:, 0:7]
y = data[:, -1]

predictions = f(X)
error = predictions - y
SSE = np.sum(error**2)
RMSE = np.sqrt(SSE / len(y))
print(RMSE)
