# -*- coding: utf-8 -*-
# 결정의 정보를 받고 구조적 특성을 맵 형태로 바꿔준다.
import numpy as np
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
import pandas as  pd
from pymatgen import Structure
from pymatgen.analysis import structure_analyzer, structure_matcher
import joblib
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import json
from tqdm import tqdm

#%% querying material project database to get cif files and material properties

# materials project의 데이터베이스를 받아오기 위해 키가 필요하다.
# 사이트에 가서 api 키를 생성할 수 있다. 한번 생성하면 계속 쓸 수 있는 것 같다.
# please use your own api key for querying MP.org
mat_api_key = '2IgbGtrUqnFHQpV9'

# materials project의 데이터베이스를 받아와서 전처리 후 해당 데이터셋을 반환
# 입력: api 키, nsites(21), least ele(2), most element(4)
def get_data (mat_api_key, nsites=41, least_ele = 1, most_ele= 6):

    mpdr = MPDataRetrieval(mat_api_key)
    # e_above_hull and formation energy per atom are the upper limit to get stable compounds
    df=mpdr.get_dataframe(criteria={'icsd_ids':{"$exists":True},
                                    'nsites':{'$lt':nsites},
                                    'e_above_hull':{'$lt':0.08},
                                    'formation_energy_per_atom':{'$lt':2},
                                    'nelements': {'$gt':least_ele,'$lt':most_ele} ,
                                    
                                     },properties=['material_id','formation_energy_per_atom','band_gap','pretty_formula','e_above_hull','elements','cif','spacegroup.number'])
    
    # df의 ind는 각 결정에 번호를 메긴 것이다. 일종의 인덱스로 사용하는 것 같다.
    df['ind'] = np.arange(0,len(df),1)
    
    # load the thermoelectric calculations dataset from the csv file
    # 열전 계산 데이터셋의 받아온다.
    df_m = pd.read_csv('df_power.csv',index_col=0)

    # 값이 없는 부분을 제거한다. (12637, 4)
    # df_m = np.log10(np.abs(df_m))
    df_m = df_m.dropna()

    # 바닥 상태의 특성과 BTE(Boltzmann Transport Equation) 계산량이 있는 물질을 고른다. (2735,)
    # 이름이 겹치는 것의 이름을 고르는 것이다.
    # select compounds that has both ground state properties and BTE calculations
    i = df.index.intersection(df_m.index)
    
    # material_id에 해당하는 물질에 mi_eff N, mi_eff P, Seebeck, Powerfactor을 추가한다.
    df_in = pd.concat([df.iloc[:,:],df_m.loc[i,:]],1)
    
#    df_in = df_in.dropna()
    
    # Seebeck 값을 양수로 바꿔준다.
    df_in['Seebeck'] = np.abs(df_in['Seebeck'] )
    # df: 8개의 특성(formation energy per atom, band_gap, pretty formula, e_above hull, elements, cif,
    #               spacegroup.number, ind)
    # df_in: df에서 4개의 특성이 추가(mi_dff N, mi_eff P, Seebeck, Powerfactor
    return df, df_in

#df1,df_in1 = get_data (mat_api_key, nsites=21, least_ele = 4, most_ele= 6)




# cif 파일을 이용해서 crystal repersentation을 만든다.
# function for featurizing crystal representation using cif files from MP.org

def  crystal_represent(df,num_ele=3,num_sites=20):

    # 피클 파일을 읽어온다. Element는 원소의 기호를 받아온다. (103,)
    Element= joblib.load('./files/element.pkl')
    # 103 원소가 있고 이를 원핫 벡터로 표현한다.(103, 103)
    E_v = np_utils.to_categorical(np.arange(0,len(Element),1))
    
    # 파일을 열어서 정보를 받아온다.
    elem_embedding_file = './files/atom_init.json'
    with open(elem_embedding_file) as f:
        # 파일 안에는 (str)1~100 키에 해당하는 92 사이즈의 배열이 있다.
        elem_embedding = json.load(f)
    # str의 키를 int 형태로 바꾼다.
    elem_embedding = {int(key): value for key, value in elem_embedding.items()}

    # (100, 92) 사이즈를 가진 배열로 바꿔준다.
    feat_cgcnn = []
    for key,value in elem_embedding.items():
        feat_cgcnn.append(value)
    feat_cgcnn = np.array(feat_cgcnn)

    # start featurization
    ftcp = []
    print('--crystal_repersent start--')
    # 26402개의 물질에 대해서 하나씩 특징화를 한다.
    for idx in tqdm(range(len(df))): #46382
        # cif로부터 구조의 특성을 얻는다. 사람이 읽을 수 있게 변환
        crystal = Structure.from_str(df['cif'][idx],fmt="cif")
        # lattice 정보만 받아온다.
        latt = crystal.lattice

        # ui는 가지고 있는 원자번호 배열(3), ux는 원소의 위치배열(3), uy는 전체에서 어떤 원소가 해당하는지에 대한 배열(16)
        ui, ux, uy = np.unique(crystal.atomic_numbers,return_index=True,return_inverse= True)
        # z_u는 원소의 주어진 순서에 맞는 배열[62, 32, 78]
        z_sorted=np.array(crystal.atomic_numbers)
        z_u = z_sorted[np.sort(ux)]
        # (원소 개수(3), 총 원소 개수(103))을 가지는 0 행렬 생성
        onehot = np.zeros((num_ele,len(E_v)))
        # [62, 32, 78]의 원소를 (3, 103) 원핫으로 각각 표현
        onehot[:len(z_u),:] = E_v[z_u-1,:]
        # (nsite(20), 3)을 가지는 0 행렬 생성
        fc1 =np.zeros((num_sites,3))
        # (nsite(20), 원소 개수(3))을 가지는 0 행렬 생성
        fc1_ind = np.zeros((num_sites,num_ele))

        # Fourier space, 1.2 is used at the maximum distance
        # latt의 정보들을 결정학적 상호격자 값으로 바꿈(회절 모양에 대한 정보)
        recip_latt = latt.reciprocal_lattice_crystallographic
        # [(fcoord, dist, index, supercell_image) …] 값으로 변환. 결정방향, 거리에 대한 정보를 가진 듯.(2333)
        recip_pts = recip_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], 1.2)


        zs = []
        coeffs = []
        fcoords = []
        coords = []
        occus = []

        # site는 원소, (x, x, x) [x, x, x] 형태ㅊ
        for site in crystal:
            # sp: 원소에 대한 정보, occu: 1.0
            for sp, occu in site.species.items():
                # 원자번호를 추가
                zs.append(sp.Z)
                # cgcnn에 들어있던 원소에 대한 정보를 받아온다.(92,)
                c = feat_cgcnn[sp.Z-1,:]
                # 원소 정보를 추가
                coeffs.append(c)
                # site의 맨뒤 배열을 가져와서 추가(frac_coords)(3,)
                fcoords.append(site.frac_coords)
                # occu인 1.0을 추가
                occus.append(occu)
                # site의 중간 배열을 가져와서 추가(coord)(3,)
                coords.append(site.coords)

        # zs(순서에 따른 원자번호(16 size)), coeffs(92개의 원소 특성(16, 92 size))
        zs = np.array(zs)
        coeffs = np.array(coeffs)

        # (요소 개수(3), 원소 특성 개수(92))의 0 행렬 생성
        coeffs_crsytal = np.zeros((num_ele,feat_cgcnn.shape[1]))

        # 원소 정보를 원소 순으로 입력(3, 92)
        coeffs_crsytal [:len(z_u),:] = coeffs[np.sort(ux),:]

        # coords에 대한 정리
        coords = np.array(coords)
        fcoords = np.array(fcoords)

        # (nsite(20), 3) -> 16개만 채운다.
        fc1[:fcoords.shape[0],:]= fcoords
        # (16) -> (16, 1)로 모양 변경
        occus = np.array(occus).reshape(-1,1)

        # 격자 상수를 불러옴(3,)
        abc1 = np.asarray(latt.abc)
        ang1 = np.asarray(latt.angles)

        # 원소가 해당하는 부분 위치에 1표시
        for i in range(len(z_u)):
             fc1_ind[np.where(z_sorted==z_u[i]),i]=1

        # (3, 22) 아까 20에서 길이, 각도 2개가 추가된 형태
        crys_list = np.concatenate((abc1.reshape(-1,1),
                                ang1.reshape(-1,1),fc1.T),axis=1)
        # 위와 같은 (3, 22) 0 행렬을 만들고 값을 채워넣는다.
        crys_list1 = np.zeros((num_ele, crys_list.shape[1]))
        crys_list1 [:crys_list.shape[0],:] = crys_list


        # real space represeatnion
        # onehot(세 원소를 원핫표현(3, 103)), crys_list1(길이, 각도, 원소별 특성(3, 22)),
        # fc1_ind(원소에 해당하는 자리에 1로 표현(20, 3)), np.zeros(3, 1),
        # coeffs_crsytal(원소 정보를 원소 순으로 입력(3, 92))
        atom_list = np.concatenate((onehot,crys_list1,fc1_ind.T,np.zeros((num_ele,1)),coeffs_crsytal),axis=1)
        # 합치고 트렌스포즈 해주면 (238, 3) 행렬 생성
        atom_list = atom_list.T

        # 실제와 푸리에 배열 생성
        hkl_list = []        
        ftcp_list = []

        #
        for hkl, g_hkl, ind, _ in sorted(recip_pts, key=lambda i: (abs(i[0][0])+abs(i[0][1])+abs(i[0][2]), -i[0][0], -i[0][1], -i[0][2])):
            # 밀러 상수를 정수로 만든다.
            # Force miller indices to be integers.
            hkl = [int(round(i)) for i in hkl]
            i += 1

            # g_hkl 값이 0이 아니고, i < 61일 때 리스트에 추가
            if g_hkl != 0 and i < 61:
                # 밀러 상수 추가
                hkl_list.append(hkl)

                # Vectorized computation of g.r for all fractional coords and
                # (16, 1) 사이즈로 fcoords(16, 3)과 hkl(1, 3)을 dot한 결과
                g_dot_r = np.dot(fcoords, np.transpose([hkl])).reshape(-1,1)
              
                # Vectorized computation.
                # (92,) 사이즈로 occus(대부분 1.0), g_dot_r(16, 1) * coeffs(16, 92) => (16, 92) => sum => (92,)
                f_hkl = np.sum((occus * np.pi * g_dot_r*coeffs),axis=0)
    #            z_hkl = np.sum(occus*g_dot_r*zs,axis=0)

                # 1번 인텍스에 g_hkl 값을 삽입(93,)
                f_hkl1 = np.insert(f_hkl, 1, g_hkl)
                # (238, 1) 사이즈로 np.zeros(145, 1), coeffs(16, 92), f_hkl1 reshape(93, 1)
                f_hkl1 = np.concatenate((np.zeros((atom_list.shape[0]-coeffs.shape[1]-1, 1)), f_hkl1.reshape(-1, 1)))
                # 계산한 (238, 1) ftcp에 추가
                ftcp_list.append(f_hkl1)              

        # Fourier space representations
        # 57, (238, 1) 배열에서 => (238, 57, 1) numpy로 변경
        ftcp_list = np.stack(ftcp_list,axis=1)
        # (238, 57)로 조정
        ftcp_list = np.squeeze(ftcp_list,axis=2)
        # atom_list(238, 3), ftcp_list(238, 57) => (238, 60)
        ftcp_list = np.concatenate((atom_list,ftcp_list),axis=1)
        # ftcp에 추가
        ftcp.append(ftcp_list)  
    
    # np 형태로 반환
    return np.stack(ftcp,axis=0)