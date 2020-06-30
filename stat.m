% Gi-Yeul Bae  2017-10-3
% ��Դ��������Bae & Luck 2018 Journal of Neuroscience��
%  ����ʵ�ּ���ƽ��׼ȷ��
%  +����cluster-based���ؿ���ģ�����
%  +��ͼ�Ĳ���
% ��������Ͻ�����������ϸע��

% ���Ա��
subList = [201 202 203 204 205 206 207 208 209 210 212 213 215 216 217 218]; 

% ��������
Nsub = length(subList);

% 3��
Nblock = 3; % cross-validation
% ��������
Nitr = 10; % iteration
% ʱ��� ��ÿ��ƽ��5��ʱ�䴰����һ��ʱ����Ӧ0.02s��
Ntp = 100; % # of time points
% 16��λ��
NBins = 16; % # of location bin

% ʱ�䷶Χ-500ms��1496ms
tm = -500:20:1496;

% ��ʼ��ƽ��׼ȷ�ʾ��� ������*ʱ�����
AverageAccuracy = nan(Nsub,Ntp);

for sub = 1:Nsub
     % ��ʼ��׼ȷ�ʾ��� ʱ�����*blocks��*������
    DecodingAccuracy = nan(Ntp,Nblock,Nitr);
    
     % ��ȡ֮ǰ��decoding���
    fileLocation = pwd;
    readThis =strcat(fileLocation,'/Location_Results_Alphabased_',num2str(subList(sub)),'.mat');
%     readThis =strcat(fileLocation,'/Orientation_Results_Alphabased_',num2str(subList(sub)),'.mat');
%     readThis =strcat(fileLocation,'/Location_Results_ERPbased_',num2str(subList(sub)),'.mat');
%     readThis =strcat(fileLocation,'/Orientation_Results_ERPbased_',num2str(subList(sub)),'.mat');
    load(readThis)
     

    % Ԥ��õ���labels
    svmPrediction = squeeze(svmECOC.modelPredict);
    % ʵ�����ݵ�labels
    tstTargets = squeeze(svmECOC.targets);
    clear svmECOC
    
    % ����ƽ���Ľ�����ȷ��
    for block = 1:Nblock
        for itr = 1:Nitr
            for tp = 1:Ntp  
                
                % ��ȡ16��������Ԥ��õ���labels
                prediction = squeeze(svmPrediction(itr,tp,block,:));
                % ��ȡ16��������ʵ�ʵ�labels
                TrueAnswer = squeeze(tstTargets(itr,tp,block,:));
                % �������
                Err = TrueAnswer - prediction;
                % ������ȷ��
                ACC = mean(Err==0);
                DecodingAccuracy(tp,block,itr) = ACC; % average decoding accuracy

            end
        end
    end
      
     % ƽ��block�͵����Ľ��
     grandAvg = squeeze(mean(mean(DecodingAccuracy,2),3));
    
     % �Խ������ʱ���ϵ�ƽ��,�����ʱ������ƽ��(t-2, t-1, t, t+1, t+2)
     smoothed = nan(1,Ntp);
     for tAvg = 1:Ntp
         if tAvg ==1
           smoothed(tAvg) = mean(grandAvg((tAvg):(tAvg+2)));
         elseif tAvg ==2
           smoothed(tAvg) = mean(grandAvg((tAvg-1):(tAvg+2)));
         elseif tAvg == (Ntp-1)
           smoothed(tAvg) = mean(grandAvg((tAvg-2):(tAvg+1)));
         elseif tAvg == Ntp
           smoothed(tAvg) = mean(grandAvg((tAvg-2):(tAvg)));
         else
           smoothed(tAvg) = mean(grandAvg((tAvg-2):(tAvg+2)));  
         end

     end
     
     % ����ƽ�������ȷ��
     AverageAccuracy(sub,:) =smoothed; % average across iteration and block
     
end

% ƽ��ÿ�����ԵĽ�����õ�ƽ������ȷ��
subAverage = squeeze(mean(AverageAccuracy,1)); 
% �����׼��
seAverage = squeeze(std(AverageAccuracy,1))/sqrt(Nsub); 

%% ��cluster mass���� 

 %��220ms-1496ms
 releventTime = 37:100;

% ��ÿ��ʱ����׼ȷ�������׼ȷ��0.0625(1/16)��t-test
Ps = nan(2,length(releventTime));
    for i = 1:length(releventTime)
        tp = releventTime(i);
 
        [H,P,CI,STATS] =  ttest(AverageAccuracy(:,tp), 0.0625, 'tail', 'right');

        Ps(1,i) = STATS.tstat;
        Ps(2,i) = P;
    end
    
% �ҵ������Եĵ�
candid = Ps(2,:) <= .05;

% �Ƴ������������Ե�
% �ж�t-1,t,t+1����ʱ���Ϊ���������������������������������ʱ���t��Ϊ������
candid_woOrphan = candid;
candid_woOrphan(1,1) = candid(1,1);
for i = 2:(length(releventTime)-1)
    if candid(1,i-1) == 0 && candid(1,i) ==1 && candid(1,i+1) ==0
    candid_woOrphan(1,i) = 0; 
    else
    candid_woOrphan(1,i) = candid(1,i);     
    end  
end

% ��ͳ�Ƶ���������Ϣƥ�䵽ʱ����
clusters = zeros(length(tm),1); % ��¼������
clusterT = zeros(length(tm),1); % ��¼tֵ
clusters(releventTime,1) = candid_woOrphan;
clusterT(releventTime,1) = Ps(1,:);
clusterTsum = sum(Ps(1,logical(candid_woOrphan))); % �ܵ�tֵ

% �ҵ��ж��ٸ�clusters��������ÿһ��cluster��Tֵ�Ӻ�
tmp = zeros(10,300); % ������¼cluster��ÿ��cluster��Ӧ��ʱ���
cl = 0; % ��¼cluster���
member = 0; % ��¼ÿһ��cluster��ʱ���ı��
for i = 2:length(clusters)-1

        % һ��cluster�Ŀ�ʼ
        if clusters(i-1) ==0 && clusters(i) == 1 && clusters(i+1) == 1 
        cl = cl+1;
        member = member +1;
        tmp(cl,member) = i;    
        
        % һ��cluster�Ľ���
        elseif clusters(i-1) ==1 && clusters(i) == 1 && clusters(i+1) == 0 
        member = member +1;  
        tmp(cl,member) = i;    
        member = 0;  
        
        % һ��cluster����
        elseif clusters(i-1) ==1 && clusters(i) == 1 && clusters(i+1) == 1             
        member = member +1;  
        tmp(cl,member) = i;    
        
        else
        
        end
end

% cluster������
HowManyClusters = cl;
a = tmp(1:cl,:); % ��ȡtmp����Ч��cluster
eachCluster = a(:,logical(sum(a,1)~=0)); % ��ȡÿ��cluster����Ч�Ĳ���

% ����ÿ��cluster��tֵ��
dat_clusterSumT = nan(HowManyClusters,1);
for c = 1:HowManyClusters
   dat_clusterSumT(c,1) = sum(clusterT(eachCluster(c,eachCluster(c,:) ~=0)));
end

%% �������ؿ���ģ��

iteration = 10000;

%% ע�⣺ģ��Ứ�ܳ�ʱ��

% ��ʼ��ģ��cluster��tֵ
SimclusterTvalue = nan(1,iteration);

for itr = 1:iteration
Ps = nan(2,length(releventTime));
    
    % ���ɼ����ݣ�ʵ�����������decoding�����labels

      simacc = nan(Nitr,Nblock);
      simaccSub = nan(Nsub,Ntp);  
      for sub = 1:Nsub
          for t = 1:Ntp 
            for it = 1:Nitr
                for blk = 1:Nblock
                simaccTest = (1:NBins) - randi(NBins,1,NBins); % sample with replacement
                simacc(it,blk) = sum(simaccTest==0)/NBins;
                end
             end
        simaccSub(sub,t) = mean(mean(simacc(:,:),1),2);
          end
      end
      
     % �ԼٵĽ������ʱ���ϵ�ƽ����ͬ������ƽ������һ��
      smtFake = nan(Nsub,Ntp);
      for tAvg = 1:Ntp
         if tAvg ==1
           smtFake(:,tAvg) = mean(simaccSub(:,(tAvg):(tAvg+2)),2);
         elseif tAvg ==2
           smtFake(:,tAvg) = mean(simaccSub(:,(tAvg-1):(tAvg+2)),2);
         elseif tAvg == (Ntp-1)
           smtFake(:,tAvg) = mean(simaccSub(:,(tAvg-2):(tAvg+1)),2);
         elseif tAvg == Ntp
           smtFake(:,tAvg) = mean(simaccSub(:,(tAvg-2):(tAvg)),2);
         else
           smtFake(:,tAvg) = mean(simaccSub(:,(tAvg-2):(tAvg+2)),2);
         end

      end
     % ����t-test    
    for i = 1:length(releventTime) 
        tp = releventTime(i);
       
        [H,P,CI,STATS] =  ttest(smtFake(:,i),0.0625, 'tail','right' );

        Ps(1,i) = STATS.tstat;
        Ps(2,i) = P;
    end
    
    % �ҵ������Ե�
    candid = Ps(2,:) <= .05;
    candid_woOrphan = zeros(1,length(candid));
    candid_woOrphan(1,1) = candid(1,1);
    candid_woOrphan(1,length(candid)) = candid(1,length(candid));
    % ȥ��������ʱ����Ϣ
    for i = 2:(length(releventTime)-1)

        if candid(1,i-1) == 0 && candid(1,i) ==1 && candid(1,i+1) ==0
        candid_woOrphan(1,i) = 0; 
        else
        candid_woOrphan(1,i) = candid(1,i);     
        end

    end

    % ��ͳ�Ƶ���������Ϣƥ�䵽ʱ����
    clusters = zeros(length(tm),1);
    clusterT = zeros(length(tm),1);
    clusters(releventTime,1) = candid_woOrphan;
    clusterT(releventTime,1) = Ps(1,:);

    % �ҵ��ж��ٸ�clusters��������ÿһ��cluster��Tֵ�Ӻ�
    tmp = zeros(50,300);
    cl = 0;
    member = 0;
    for i = 2:length(clusters)-1

            if clusters(i-1) ==0 && clusters(i) == 1 && clusters(i+1) == 1 
            cl = cl+1;
            member = member +1;
            tmp(cl,member) = i;    

            elseif clusters(i-1) ==1 && clusters(i) == 1 && clusters(i+1) == 0 
            member = member +1;  
            tmp(cl,member) = i;    
            member = 0;  
            
            elseif clusters(i-1) ==1 && clusters(i) == 1 && clusters(i+1) == 1             
            member = member +1;  
            tmp(cl,member) = i;    
            
            else
                
            end
    end

    HowManyClusters = cl;
    if HowManyClusters >0
    a = tmp(1:cl,:);
    sim_eachCluster = a(:,logical(sum(a,1)~=0));

    % ����ÿ��cluster��tֵ��
    sim_clusterSumT = zeros(HowManyClusters,1);
    for c = 1:HowManyClusters
       sim_clusterSumT(c,1) = sum(clusterT(sim_eachCluster(c,sim_eachCluster(c,:) ~=0)));
    end

    % �ҵ�����ֵ����Tֵ
    record = abs(sim_clusterSumT) == max(abs(sim_clusterSumT));
    SimclusterTvalue(1,itr) = sim_clusterSumT(record);
    
    % ���û������cluster��Tֵ��Ϊ0
    else
    SimclusterTvalue(1,itr) = 0;    
    end
        
end % ģ�����

% ��10000�ε�����Tֵ����
sortedTvalues = sort(SimclusterTvalue,2);

%% �ҵ�95%���ٽ�t
cutOff = iteration - iteration * 0.05;
critT = sortedTvalues(cutOff);
% ��ȡ��ʵdecoding��cluster tֵ����
sigCluster = dat_clusterSumT > critT;


%% ��ͼ

figure(1)
cl=colormap(parula(50));

accEst = squeeze(subAverage);  
draw = eachCluster(sigCluster,:);
draw = sort(reshape(draw,1,size(draw,1)*size(draw,2)));
draw = draw(draw>0);

w = zeros(Ntp,1);
w(draw)=1;
a = area(1:length(tm), accEst.*w');
a.EdgeColor = 'none';
a.FaceColor = [0.8,0.8,0.8];
child = get(a,'Children');
set(child,'FaceAlpha',0.9)
hold on
mEI = boundedline(1:length(tm),subAverage,seAverage, 'cmap',cl(42,:),'alpha','transparency',0.35);
xlabel('Time (ms)');ylabel('Decoding Accuracy')
ax = gca;
ax.YLim = [0.05, 0.15];
ax.YTick = [0,0.02,0.04,0.06,0.08,0.10,0.12,0.14];
ax.XTick = [1 26 51 76 100]; 
ax.XTickLabel = {'-500','0','500','1000','1500'};
h = line(1:length(tm),0.0625* ones(1,Ntp));
h.LineStyle = '--';
h.Color = [0.1,0.1,0.1];
hold off
