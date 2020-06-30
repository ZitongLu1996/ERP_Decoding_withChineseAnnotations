% ��Դ��������Bae & Luck 2018 Journal of Neuroscience��
% Experiment 2����ʵ��ERP Decoding�Ĳ���
% ��������Ͻ�����������ϸע��

function SVM_ECOC_ERP_Decoding(subs)

% �رղ��м���
delete(gcp)
parpool

if nargin ==0
    % ����id
    subs = [201 202 203 204 205 206 207 208 209 210 212 213 215 216 217 218];       

end

% ��ȡ��������
nSubs = length(subs);


%% ���ñ���

% �����������
svmECOC.nChans = 16; 
svmECOC.nBins = svmECOC.nChans; 
% ��������
svmECOC.nIter = 10;
% ������֤�õ�Blocks��
svmECOC.nBlocks = 3; 
% ��ͨ�˲�����
svmECOC.frequencies = [0 6];
% ���ڷ�����ʱ���
% -500ms��1496ms
% ���20msһ�β���
svmECOC.time = -500:20:1496; 
% 1��ʱ������1+4�������㣨ǰ���2����ʱ�䴰
svmECOC.window = 4;
% ������
svmECOC.Fs = 250;
% 16�����ڷ����ĵ�����Ӧ�����еı��
ReleventChan = sort([2,3,4,18,19, 5,6,20, 7,8,21, 9,10,11,12,13,14, 22,23,24,25,26,27, 15,16,1,17]);
% ��ص���������
svmECOC.nElectrodes = length(ReleventChan);


%% �򻯱���

nChans = svmECOC.nChans;
nBins = svmECOC.nBins;
nIter = svmECOC.nIter;
nBlocks = svmECOC.nBlocks;
freqs = svmECOC.frequencies;
times = svmECOC.time;
nElectrodes = svmECOC.nElectrodes;
nSamps = length(svmECOC.time);
Fs = svmECOC.Fs;

%% ������������
for cond = 1:2
    % 1����λ��, 2������
    
%% �������б���
for s = 1:nSubs
    % ��ȡ��s�����Եı��
    sn = subs(s);

    fprintf('Subject:\t%d\n',sn)

    %% ��ȡ����
    currentSub = num2str(sn); % ���Ա�ŵ����ͱ���ַ���
    dataLocation = pwd; % �������ݴ洢��·��
    % �����ĸñ��Ե����ݵ�ַ
    loadThis = strcat(dataLocation,'/Decoding_OL_',currentSub,'.mat');
    % ��������
    load(loadThis)
    
    saveLocation = pwd; % ���ô洢·��
    
    %% ����ÿһ���Դε�λ��/�������Ϣ
    
    if cond ==1
    channel = data.targetLocBin; % ��ȡ���ݵ�λ����Ϣ
    else
    channel = data.targetOriBin; % ��ȡ���ݵĳ�����Ϣ
    end
    
    % ��channelת�ò���ΪposBin��Ϊ������Ϣ
    svmECOC.posBin = channel';
    posBin = svmECOC.posBin;
    
    % ��ȡ���ڷ�����16��������eeg����
    eegs = data.eeg(:,ReleventChan,:); 
    
    %% ����ʱ���
    
    % �����ݼ�¼��ʱ�������Ҫ���ڷ�����ʱ�����ƥ��
    % ����ʱ�����������ƥ���ʱ���Ϊ1����ƥ���ʱ���Ϊ0
    tois = ismember(data.time.pre:4:data.time.post,svmECOC.time); 
    % ����ʱ��������
    nTimes = length(tois);
    
    % �Դ���
    svmECOC.nTrials = length(posBin);
    nTrials = svmECOC.nTrials; 

    %% Ԥ�����

    % �����洢SVM��Ԥ�����ľ���svm_predict
    % [��������, ������, block��, ������]
    svm_predict = nan(nIter,nSamps,nBlocks,nChans);
    % �����洢��ʵ��Ŀ��ֵ�ľ���tst_target
    tst_target = nan(nIter,nSamps,nBlocks,nChans);  % a matrix to save true target values
    % �洢block�ķ�����Ϣ
    svmECOC.blocks = nan(nTrials,nIter);  % create svmECOC.block to save block assignments


    %% ��ͨ�˲�
    
    % ��ʼ�����������洢�˲��������
    filtData = nan(nTrials,nElectrodes,nTimes);
    % ���ж�ÿһ�����������˲�
    parfor c = 1:nElectrodes            
          filtData(:,c,:) = eegfilt(squeeze(eegs(:,c,:)),Fs,freqs(1,1),freqs(1,2)); % low pass filter
    end

    %% ��������
    
    tic % ��ʼ��ʱ

    for iter = 1:nIter

        % Ԥ�����������飬����Ϊ�Դ���
        blocks = nan(size(posBin));
        shuffBlocks = nan(size(posBin));

        clear binCnt

        % ȷ��ÿһ���������Դ���
        for bin = 1:nBins
            binCnt(bin) = sum(posBin == bin); 
        end

        % ��С�������Դ���
        minCnt = min(binCnt);
        % ÿһ��block����С�������Դ���������ȡ����
        nPerBin = floor(minCnt/nBlocks);


        % �����Դ�
        shuffInd = randperm(nTrials)'; % ��������Դ�id
        shuffBin = posBin(shuffInd); % ����Դζ�Ӧ��������Ϣ

        %% ��ȡÿһ���������Դ�����
        %% ����ÿ�������Դ�һ�µ�label
        
        for bin = 1:nBins   
            % ��ȡ����bin��Ӧ���Դ�λ��
            idx = find(shuffBin == bin); 
            % �޳�������Դ�
            % ֻ����nPerBin*nBlocks��ô����Դ�
            % ����Ϊ�˱�֤ÿһ���������Դ���һ��
            idx = idx(1:nPerBin*nBlocks);

            % ����[1,nBlocks]����ͬ������labels
            x = repmat((1:nBlocks)',nPerBin,1);
            % ��label�����Ӧ���Դ�
            shuffBlocks(idx) = x;

        end

        % ��δ���ҵ���Ϣ����blocks����
        blocks(shuffInd) = shuffBlocks;
        svmECOC.blocks(:,iter) = blocks; % block assignment
        svmECOC.nTrialsPerBlock = length(blocks(blocks == 1));

        % Average data for each position bin across blocks
        
        posBins = 1:nBins;
        % ����һ�����������洢���˲������ݽ���50hz�ز����������
        blockDat_filtData = nan(nBins*nBlocks,nElectrodes,nSamps);
        % ����һ���������ڴ洢labels
        labels = nan(nBins*nBlocks,1);
        % ����һ���������ڴ洢block���
        blockNum = nan(nBins*nBlocks,1);                                % block numbers for averaged & filtered EEG data

        bCnt = 1;

        % ��blockDat_filtData����������
        for ii = 1:nBins
            for iii = 1:nBlocks
                blockDat_filtData(bCnt,:,:) = squeeze(mean(filtData(posBin==posBins(ii) & blocks==iii,:,tois),1));
                labels(bCnt) = ii;
                blockNum(bCnt) = iii;
                bCnt = bCnt+1;
            end
        end

        %% ��ÿ��ʱ������SVM����
        
        parfor t = 1:nSamps

            % ��ȡʱ���t������
            % �����t�ǰ��ղ��������ʵ��ʱ��ȡ
            % ��ȡ����һ��ʱ�䴰�����ݣ�5��ʱ��㣩
            toi = ismember(times,times(t)-svmECOC.window/2:times(t)+svmECOC.window/2);
            
            % ƽ��ʱ��������
            dataAtTimeT = squeeze(mean(blockDat_filtData(:,:,toi),3));  

            % ��ÿһ��block����
            % ÿһ��block����Ϊһ�β��Լ�
            for i=1:nBlocks
                % ѵ��labels
                trnl = labels(blockNum~=i);
                % ����labels
                tstl = labels(blockNum==i);
                % ѵ������
                trnD = dataAtTimeT(blockNum~=i,:);
                % ��������
                tstD = dataAtTimeT(blockNum==i,:);    % test data

                % ѵ��SVM
                mdl = fitcecoc(trnD,trnl, 'Coding','onevsall','Learners','SVM' );   %train support vector mahcine
                % ��ѵ���õķ���������Ԥ�⣨���ԣ�
                LabelPredicted = predict(mdl, tstD);
                % ����Ԥ��õ���labels
                svm_predict(iter,t,i,:) = LabelPredicted;
                % ����ʵ�����ݵ�labels
                tst_target(iter,t,i,:) = tstl;

            end

        end

    end

    % ������ʱ
    toc
    
    % �洢��λ��decoding�Ľ��
    if cond ==1
    OutputfName = strcat(saveLocation,'/Location_Results_ERPbased_',currentSub,'.mat');
    % �洢�Գ���decoding�Ľ��
    elseif cond ==2
    OutputfName = strcat(saveLocation,'/Orientation_Results_ERPbased_',currentSub,'.mat');
    end
    svmECOC.targets = tst_target;
    svmECOC.modelPredict = svm_predict; 

    svmECOC.nBlocks = nBlocks;

    save(OutputfName,'svmECOC','-v7.3');

end
end
