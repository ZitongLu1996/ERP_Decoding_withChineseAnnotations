% 该源码来自于Bae & Luck 2018 Journal of Neuroscience的
% Experiment 2用于实现ERP Decoding的部分
% 在其基础上进行了中文详细注解

function SVM_ECOC_ERP_Decoding(subs)

% 关闭并行计算
delete(gcp)
parpool

if nargin ==0
    % 被试id
    subs = [201 202 203 204 205 206 207 208 209 210 212 213 215 216 217 218];       

end

% 获取被试数量
nSubs = length(subs);


%% 设置变量

% 分类的条件数
svmECOC.nChans = 16; 
svmECOC.nBins = svmECOC.nChans; 
% 迭代次数
svmECOC.nIter = 10;
% 交叉验证用的Blocks数
svmECOC.nBlocks = 3; 
% 低通滤波参数
svmECOC.frequencies = [0 6];
% 用于分析的时间段
% -500ms到1496ms
% 间隔20ms一次采样
svmECOC.time = -500:20:1496; 
% 1个时间点代表1+4个采样点（前后各2）的时间窗
svmECOC.window = 4;
% 采样率
svmECOC.Fs = 250;
% 16个用于分析的导联对应数据中的编号
ReleventChan = sort([2,3,4,18,19, 5,6,20, 7,8,21, 9,10,11,12,13,14, 22,23,24,25,26,27, 15,16,1,17]);
% 相关导联的数量
svmECOC.nElectrodes = length(ReleventChan);


%% 简化变量

nChans = svmECOC.nChans;
nBins = svmECOC.nBins;
nIter = svmECOC.nIter;
nBlocks = svmECOC.nBlocks;
freqs = svmECOC.frequencies;
times = svmECOC.time;
nElectrodes = svmECOC.nElectrodes;
nSamps = length(svmECOC.time);
Fs = svmECOC.Fs;

%% 遍历两种条件
for cond = 1:2
    % 1代表位置, 2代表朝向
    
%% 遍历所有被试
for s = 1:nSubs
    % 提取第s个被试的编号
    sn = subs(s);

    fprintf('Subject:\t%d\n',sn)

    %% 读取数据
    currentSub = num2str(sn); % 被试编号的类型变成字符串
    dataLocation = pwd; % 设置数据存储的路径
    % 完整的该被试的数据地址
    loadThis = strcat(dataLocation,'/Decoding_OL_',currentSub,'.mat');
    % 加载数据
    load(loadThis)
    
    saveLocation = pwd; % 设置存储路径
    
    %% 设置每一个试次的位置/朝向的信息
    
    if cond ==1
    channel = data.targetLocBin; % 获取数据的位置信息
    else
    channel = data.targetOriBin; % 获取数据的朝向信息
    end
    
    % 对channel转置并存为posBin作为特征信息
    svmECOC.posBin = channel';
    posBin = svmECOC.posBin;
    
    % 提取用于分析的16个导联的eeg数据
    eegs = data.eeg(:,ReleventChan,:); 
    
    %% 设置时间点
    
    % 将数据记录的时间点与需要用于分析的时间点做匹配
    % 数据时间点序列上有匹配的时间点为1，无匹配的时间点为0
    tois = ismember(data.time.pre:4:data.time.post,svmECOC.time); 
    % 数据时间点的数量
    nTimes = length(tois);
    
    % 试次数
    svmECOC.nTrials = length(posBin);
    nTrials = svmECOC.nTrials; 

    %% 预设矩阵

    % 用来存储SVM的预测结果的矩阵svm_predict
    % [迭代次数, 样本数, block数, 导联数]
    svm_predict = nan(nIter,nSamps,nBlocks,nChans);
    % 用来存储真实的目标值的矩阵tst_target
    tst_target = nan(nIter,nSamps,nBlocks,nChans);  % a matrix to save true target values
    % 存储block的分配信息
    svmECOC.blocks = nan(nTrials,nIter);  % create svmECOC.block to save block assignments


    %% 低通滤波
    
    % 初始化矩阵用来存储滤波后的数据
    filtData = nan(nTrials,nElectrodes,nTimes);
    % 并行对每一个导联进行滤波
    parfor c = 1:nElectrodes            
          filtData(:,c,:) = eegfilt(squeeze(eegs(:,c,:)),Fs,freqs(1,1),freqs(1,2)); % low pass filter
    end

    %% 遍历迭代
    
    tic % 开始计时

    for iter = 1:nIter

        % 预设置两个数组，长度为试次数
        blocks = nan(size(posBin));
        shuffBlocks = nan(size(posBin));

        clear binCnt

        % 确认每一种条件的试次数
        for bin = 1:nBins
            binCnt(bin) = sum(posBin == bin); 
        end

        % 最小的条件试次数
        minCnt = min(binCnt);
        % 每一个block的最小的条件试次数（向下取整）
        nPerBin = floor(minCnt/nBlocks);


        % 打乱试次
        shuffInd = randperm(nTrials)'; % 随机排列试次id
        shuffBin = posBin(shuffInd); % 随后试次对应的特征信息

        %% 获取每一种条件的试次数据
        %% 生成每种条件试次一致的label
        
        for bin = 1:nBins   
            % 获取条件bin对应的试次位置
            idx = find(shuffBin == bin); 
            % 剔除多余的试次
            % 只保留nPerBin*nBlocks这么多的试次
            % 这是为了保证每一种条件的试次数一致
            idx = idx(1:nPerBin*nBlocks);

            % 生成[1,nBlocks]个相同条件的labels
            x = repmat((1:nBlocks)',nPerBin,1);
            % 把label传入对应的试次
            shuffBlocks(idx) = x;

        end

        % 将未打乱的信息存于blocks矩阵
        blocks(shuffInd) = shuffBlocks;
        svmECOC.blocks(:,iter) = blocks; % block assignment
        svmECOC.nTrialsPerBlock = length(blocks(blocks == 1));

        % Average data for each position bin across blocks
        
        posBins = 1:nBins;
        % 定义一个矩阵用来存储对滤波后数据进行50hz重采样后的数据
        blockDat_filtData = nan(nBins*nBlocks,nElectrodes,nSamps);
        % 定义一个矩阵用于存储labels
        labels = nan(nBins*nBlocks,1);
        % 定义一个矩阵用于存储block标号
        blockNum = nan(nBins*nBlocks,1);                                % block numbers for averaged & filtered EEG data

        bCnt = 1;

        % 向blockDat_filtData矩阵传入数据
        for ii = 1:nBins
            for iii = 1:nBlocks
                blockDat_filtData(bCnt,:,:) = squeeze(mean(filtData(posBin==posBins(ii) & blocks==iii,:,tois),1));
                labels(bCnt) = ii;
                blockNum(bCnt) = iii;
                bCnt = bCnt+1;
            end
        end

        %% 对每个时间点进行SVM分类
        
        parfor t = 1:nSamps

            % 提取时间点t的数据
            % 这里的t是按照采样点而非实际时间取
            % 提取的是一个时间窗的数据（5个时间点）
            toi = ismember(times,times(t)-svmECOC.window/2:times(t)+svmECOC.window/2);
            
            % 平均时间点的数据
            dataAtTimeT = squeeze(mean(blockDat_filtData(:,:,toi),3));  

            % 对每一个block迭代
            % 每一个block都作为一次测试集
            for i=1:nBlocks
                % 训练labels
                trnl = labels(blockNum~=i);
                % 测试labels
                tstl = labels(blockNum==i);
                % 训练数据
                trnD = dataAtTimeT(blockNum~=i,:);
                % 测试数据
                tstD = dataAtTimeT(blockNum==i,:);    % test data

                % 训练SVM
                mdl = fitcecoc(trnD,trnl, 'Coding','onevsall','Learners','SVM' );   %train support vector mahcine
                % 用训练好的分类器进行预测（测试）
                LabelPredicted = predict(mdl, tstD);
                % 保存预测得到的labels
                svm_predict(iter,t,i,:) = LabelPredicted;
                % 保存实际数据的labels
                tst_target(iter,t,i,:) = tstl;

            end

        end

    end

    % 结束计时
    toc
    
    % 存储对位置decoding的结果
    if cond ==1
    OutputfName = strcat(saveLocation,'/Location_Results_ERPbased_',currentSub,'.mat');
    % 存储对朝向decoding的结果
    elseif cond ==2
    OutputfName = strcat(saveLocation,'/Orientation_Results_ERPbased_',currentSub,'.mat');
    end
    svmECOC.targets = tst_target;
    svmECOC.modelPredict = svm_predict; 

    svmECOC.nBlocks = nBlocks;

    save(OutputfName,'svmECOC','-v7.3');

end
end
