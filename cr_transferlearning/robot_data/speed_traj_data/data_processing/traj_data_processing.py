import numpy as np
from itertools import islice

#想增加数据多样性的话：
# for i in range(data1.shape[0]):
#     if i % 2 == 0:
#         data2.append(data1[i,:,:])

def load_and_process_data(data):
    data1 = []
    for j in range(len(data) - 14):
        data1.append(data[j:j + 15, :])
    data1 = np.array(data1)
    data1 = np.transpose(data1, (1, 0, 2))
    return data1

udim = 5
if udim==7:
    def train_data(segment_data):
        valid_segments = []
        for segment_name, segment_i_data in segment_data.items():
            if segment_i_data.shape[0] >= 16:
                valid_segments.append((segment_name, segment_i_data))

        numbered_segments = [(index, segment_i_data) for index, (segment_name, segment_i_data) in
                             enumerate(valid_segments)]

        Ktrain_data = []
        Ktest_data = []

        dividing_num = int(3 * len(numbered_segments) / 4)
        print(dividing_num)

        train_segments = list(islice(numbered_segments, dividing_num))
        train_segments = [segment_i_data for index, segment_i_data in train_segments]
        for i in range(len(train_segments)):
            data = np.concatenate((train_segments[i][:, :7], train_segments[i][:, 9:16], train_segments[i][:, 18:25]),
                                  axis=1)
            data = load_and_process_data(data)
            Ktrain_data.append(data)
        Ktrain_data = np.concatenate(Ktrain_data, axis=1)


        Ktest_segments = list(islice(numbered_segments, dividing_num, len(numbered_segments) + 1))
        Ktest_segments = [segment[1] for segment in Ktest_segments]
        for i in range(len(Ktest_segments)):
            data = np.concatenate((Ktest_segments[i][:, :7], Ktest_segments[i][:, 9:16], Ktest_segments[i][:, 18:25]),
                                  axis=1)
            data = load_and_process_data(data)
            Ktest_data.append(data)
        Ktest_data = np.concatenate(Ktest_data, axis=1)
        return Ktrain_data,Ktest_data


    file_paths = [
        '/home/ccr/project-koopman/robot_data/speed_traj_data/traj_origin_file/all_segment_data_franka.npy',
        '/home/ccr/project-koopman/robot_data/speed_traj_data/traj_origin_file/all_segment_testdata2_franka.npy',
    ]

    segment_data_list = [np.load(file_path, allow_pickle=True).item() for file_path in file_paths]

    # 对每个数据集进行训练和测试数据的分割
    Ktrain_data_list = []
    Ktest_data_list = []
    for segment_data in segment_data_list:
        Ktrain_data, Ktest_data = train_data(segment_data)
        Ktrain_data_list.append(Ktrain_data)
        Ktest_data_list.append(Ktest_data)

    # 合并所有训练和测试数据
    Ktrain_data = np.concatenate(Ktrain_data_list, axis=1)[:,:20000,:]
    Ktest_data = np.concatenate(Ktest_data_list, axis=1)[:,:7000,:]
    print(Ktrain_data.shape)
    print(Ktest_data.shape)
    np.save('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/Ktrain_data_franka.npy',Ktrain_data)
    np.save('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/Ktest_data_franka.npy',Ktest_data)
    Kmeasure_data = np.concatenate(Ktrain_data_list, axis=1)[:, 20000:, :]
    np.save('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/Kmeasure_data_franka.npy', Kmeasure_data)

    segment_data = np.load('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_origin_file/all_segment_testdata_franka.npy', allow_pickle=True).item()
    test_segments = list(islice(segment_data.items(), 1))
    test_segments = [segment[1] for segment in test_segments]
    for i in range(len(test_segments)):
        data = np.concatenate((test_segments[i][:, :7], test_segments[i][:, 9:16], test_segments[i][:, 18:25]), axis=1)
        np.save('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data1_franka.npy',data)
        print(data.shape)
    test_segments = list(islice(segment_data.items(), 1,2))
    test_segments = [segment[1] for segment in test_segments]
    for i in range(len(test_segments)):
        data = np.concatenate((test_segments[i][:, :7], test_segments[i][:, 9:16], test_segments[i][:, 18:25]), axis=1)
        np.save('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data2_franka.npy',data)
        print(data.shape)
    test_segments = list(islice(segment_data.items(), 2, 3))
    test_segments = [segment[1] for segment in test_segments]
    for i in range(len(test_segments)):
        data = np.concatenate((test_segments[i][:, :7], test_segments[i][:, 9:16], test_segments[i][:, 18:25]), axis=1)
        np.save('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data3_franka.npy',data)
        print(data.shape)
    test_segments = list(islice(segment_data.items(), 3, 4))
    test_segments = [segment[1] for segment in test_segments]
    for i in range(len(test_segments)):
        data = np.concatenate((test_segments[i][:, :7], test_segments[i][:, 9:16], test_segments[i][:, 18:25]), axis=1)
        np.save('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data4_franka.npy',data)
        print(data.shape)

elif udim == 6:
    def train_data(segment_data):
        valid_segments = []
        for segment_name, segment_i_data in segment_data.items():
            if segment_i_data.shape[0] >= 16:
                valid_segments.append((segment_name, segment_i_data))

        numbered_segments = [(index, segment_i_data) for index, (segment_name, segment_i_data) in
                             enumerate(valid_segments)]


        Ktrain_data = []
        Ktest_data = []

        dividing_num = int(3 * len(numbered_segments) / 4)
        print(dividing_num)

        train_segments = list(islice(numbered_segments, dividing_num))
        train_segments = [segment_i_data for index, segment_i_data in train_segments]
        for i in range(len(train_segments)):
            data = load_and_process_data(train_segments[i])
            Ktrain_data.append(data)
        Ktrain_data = np.concatenate(Ktrain_data, axis=1)

        Ktest_segments = list(islice(numbered_segments, dividing_num, len(numbered_segments) + 1))
        Ktest_segments = [segment[1] for segment in Ktest_segments]
        for i in range(len(Ktest_segments)):
            data = load_and_process_data(train_segments[i])
            Ktest_data.append(data)
        Ktest_data = np.concatenate(Ktest_data, axis=1)
        return Ktrain_data, Ktest_data


    file_paths = [
        '/home/ccr/project-koopman/robot_data/speed_traj_data/traj_origin_file/all_segment_data_ur.npy',
        '/home/ccr/project-koopman/robot_data/speed_traj_data/traj_origin_file/all_segment_testdata2_ur.npy',
    ]

    segment_data_list = [np.load(file_path, allow_pickle=True).item() for file_path in file_paths]

    # 对每个数据集进行训练和测试数据的分割
    Ktrain_data_list = []
    Ktest_data_list = []
    for segment_data in segment_data_list:
        Ktrain_data, Ktest_data = train_data(segment_data)
        Ktrain_data_list.append(Ktrain_data)
        Ktest_data_list.append(Ktest_data)

    # 合并所有训练和测试数据
    Ktrain_data = np.concatenate(Ktrain_data_list, axis=1)[:,:20000,:]
    Ktest_data = np.concatenate(Ktest_data_list, axis=1)[:,:7000,:]
    print(Ktrain_data.shape)
    print(Ktest_data.shape)
    np.save('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/Ktrain_data_ur.npy',Ktrain_data)
    np.save('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/Ktest_data_ur.npy',Ktest_data)
    Kmeasure_data = np.concatenate(Ktrain_data_list, axis=1)[:, 20000:, :]
    np.save('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/Kmeasure_data_ur.npy', Kmeasure_data)

    segment_data = np.load('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_origin_file/all_segment_testdata_ur.npy',allow_pickle=True).item()
    test_segments = list(islice(segment_data.items(), 1))
    test_segments = [segment[1] for segment in test_segments]
    for i in range(len(test_segments)):
        data = test_segments[i]
        np.save('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data1_ur.npy',data)
        print(data.shape)
    test_segments = list(islice(segment_data.items(), 1, 2))
    test_segments = [segment[1] for segment in test_segments]
    for i in range(len(test_segments)):
        data = test_segments[i]
        np.save('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data2_ur.npy',data)
        print(data.shape)
    test_segments = list(islice(segment_data.items(), 2, 3))
    test_segments = [segment[1] for segment in test_segments]
    for i in range(len(test_segments)):
        data = test_segments[i]
        np.save('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data3_ur.npy',data)
        print(data.shape)
    test_segments = list(islice(segment_data.items(), 3, 4))
    test_segments = [segment[1] for segment in test_segments]
    for i in range(len(test_segments)):
        data = test_segments[i]
        np.save('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data4_ur.npy',data)
        print(data.shape)

else:
    def train_data(segment_data):
        valid_segments = []
        for segment_name, segment_i_data in segment_data.items():
            if segment_i_data.shape[0] >= 16:
                valid_segments.append((segment_name, segment_i_data))

        numbered_segments = [(index, segment_i_data) for index, (segment_name, segment_i_data) in
                             enumerate(valid_segments)]


        Ktrain_data = []
        Ktest_data = []

        dividing_num = int(3 * len(numbered_segments) / 4)
        print(dividing_num)

        train_segments = list(islice(numbered_segments, dividing_num))
        train_segments = [segment_i_data for index, segment_i_data in train_segments]
        for i in range(len(train_segments)):
            data = np.concatenate((train_segments[i][:, :5], train_segments[i][:, 11:16], train_segments[i][:, 22:27]),
                                  axis=1)
            data = load_and_process_data(data)
            Ktrain_data.append(data)
        Ktrain_data = np.concatenate(Ktrain_data, axis=1)


        Ktest_segments = list(islice(numbered_segments, dividing_num, len(numbered_segments) + 1))
        Ktest_segments = [segment[1] for segment in Ktest_segments]
        for i in range(len(Ktest_segments)):
            data = np.concatenate((Ktest_segments[i][:, :5], Ktest_segments[i][:, 11:16], Ktest_segments[i][:, 22:27]),
                                  axis=1)
            data = load_and_process_data(data)
            Ktest_data.append(data)
        Ktest_data = np.concatenate(Ktest_data, axis=1)
        return Ktrain_data,Ktest_data


    file_paths = [
        '/home/ccr/project-koopman/robot_data/speed_traj_data/traj_origin_file/all_segment_data_dofbot.npy',
        '/home/ccr/project-koopman/robot_data/speed_traj_data/traj_origin_file/all_segment_testdata2_dofbot.npy'
    ]

    segment_data_list = [np.load(file_path, allow_pickle=True).item() for file_path in file_paths]

    # 对每个数据集进行训练和测试数据的分割
    Ktrain_data_list = []
    Ktest_data_list = []
    for segment_data in segment_data_list:
        Ktrain_data, Ktest_data = train_data(segment_data)
        Ktrain_data_list.append(Ktrain_data)
        Ktest_data_list.append(Ktest_data)

    # 合并所有训练和测试数据
    Ktrain_data = np.concatenate(Ktrain_data_list, axis=1)[:,:20000,:]
    Ktest_data = np.concatenate(Ktest_data_list, axis=1)[:,:7000,:]
    print(Ktrain_data.shape)
    print(Ktest_data.shape)
    np.save('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/Ktrain_data_dofbot.npy',Ktrain_data)
    np.save('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/Ktest_data_dofbot.npy',Ktest_data)
    Kmeasure_data = np.concatenate(Ktrain_data_list, axis=1)[:, 20000:, :]
    np.save('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/Kmeasure_data_dofbot.npy', Kmeasure_data)


    segment_data = np.load('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_origin_file/all_segment_testdata_dofbot.npy',allow_pickle=True).item()
    test_segments = list(islice(segment_data.items(), 1))
    test_segments = [segment[1] for segment in test_segments]
    for i in range(len(test_segments)):
        data = np.concatenate((test_segments[i][:, :5], test_segments[i][:, 11:16], test_segments[i][:, 22:27]), axis=1)
        np.save('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data1_dofbot.npy',data)
        print(data.shape)
    test_segments = list(islice(segment_data.items(), 1, 2))
    test_segments = [segment[1] for segment in test_segments]
    for i in range(len(test_segments)):
        data = np.concatenate((test_segments[i][:, :5], test_segments[i][:, 11:16], test_segments[i][:, 22:27]), axis=1)
        np.save('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data2_dofbot.npy',data)
        print(data.shape)
    test_segments = list(islice(segment_data.items(), 2, 3))
    test_segments = [segment[1] for segment in test_segments]
    for i in range(len(test_segments)):
        data = np.concatenate((test_segments[i][:, :5], test_segments[i][:, 11:16], test_segments[i][:, 22:27]), axis=1)
        np.save('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data3_dofbot.npy',data)
        print(data.shape)
    test_segments = list(islice(segment_data.items(), 3, 4))
    test_segments = [segment[1] for segment in test_segments]
    for i in range(len(test_segments)):
        data = np.concatenate((test_segments[i][:, :5], test_segments[i][:, 11:16], test_segments[i][:, 22:27]), axis=1)
        np.save('/home/ccr/project-koopman/robot_data/speed_traj_data/traj_train_file/test_data4_dofbot.npy',data)
        print(data.shape)
