% Gi-Yeul Bae  2017-10-3
% 该源码来自于Bae & Luck 2018 Journal of Neuroscience的
%  用于实现计算平均准确率
%  +进行cluster-based蒙特卡洛模拟分析
%  +画图的部分
% 在其基础上进行了中文详细注解

% 被试编号
subList = [201 202 203 204 205 206 207 208 209 210 212 213 215 216 217 218]; 

% 被试数量
Nsub = length(subList);

% 3折
Nblock = 3; % cross-validation
% 迭代次数
Nitr = 10; % iteration
% 时间点 （每次平均5个时间窗，即一个时间点对应0.02s）
Ntp = 100; % # of time points
% 16个位置
NBins = 16; % # of location bin

% 时间范围-500ms到1496ms
tm = -500:20:1496;

% 初始化平均准确率矩阵 被试数*时间点数
AverageAccuracy = nan(Nsub,Ntp);

for sub = 1:Nsub
     % 初始化准确率矩阵 时间点数*blocks数*迭代数
    DecodingAccuracy = nan(Ntp,Nblock,Nitr);
    
     % 读取之前的decoding结果
    fileLocation = pwd;
    readThis =strcat(fileLocation,'/Location_Results_Alphabased_',num2str(subList(sub)),'.mat');
%     readThis =strcat(fileLocation,'/Orientation_Results_Alphabased_',num2str(subList(sub)),'.mat');
%     readThis =strcat(fileLocation,'/Location_Results_ERPbased_',num2str(subList(sub)),'.mat');
%     readThis =strcat(fileLocation,'/Orientation_Results_ERPbased_',num2str(subList(sub)),'.mat');
    load(readThis)
     

    % 预测得到的labels
    svmPrediction = squeeze(svmECOC.modelPredict);
    % 实际数据的labels
    tstTargets = squeeze(svmECOC.targets);
    clear svmECOC
    
    % 计算平均的解码正确率
    for block = 1:Nblock
        for itr = 1:Nitr
            for tp = 1:Ntp  
                
                % 获取16个条件下预测得到的labels
                prediction = squeeze(svmPrediction(itr,tp,block,:));
                % 获取16个条件下实际的labels
                TrueAnswer = squeeze(tstTargets(itr,tp,block,:));
                % 计算误差
                Err = TrueAnswer - prediction;
                % 计算正确率
                ACC = mean(Err==0);
                DecodingAccuracy(tp,block,itr) = ACC; % average decoding accuracy

            end
        end
    end
      
     % 平均block和迭代的结果
     grandAvg = squeeze(mean(mean(DecodingAccuracy,2),3));
    
     % 对结果进行时间上的平滑,对五个时间点进行平均(t-2, t-1, t, t+1, t+2)
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
     
     % 保存平滑后的正确率
     AverageAccuracy(sub,:) =smoothed; % average across iteration and block
     
end

% 平均每个被试的结果，得到平均的正确率
subAverage = squeeze(mean(AverageAccuracy,1)); 
% 计算标准误
seAverage = squeeze(std(AverageAccuracy,1))/sqrt(Nsub); 

%% 做cluster mass分析 

 %从220ms-1496ms
 releventTime = 37:100;

% 对每个时间点的准确率与随机准确率0.0625(1/16)做t-test
Ps = nan(2,length(releventTime));
    for i = 1:length(releventTime)
        tp = releventTime(i);
 
        [H,P,CI,STATS] =  ttest(AverageAccuracy(:,tp), 0.0625, 'tail', 'right');

        Ps(1,i) = STATS.tstat;
        Ps(2,i) = P;
    end
    
% 找到显著性的点
candid = Ps(2,:) <= .05;

% 移除孤立的显著性点
% 判断t-1,t,t+1三个时间点为不显著，显著，不显著的情况，将这种时候的t设为不显著
candid_woOrphan = candid;
candid_woOrphan(1,1) = candid(1,1);
for i = 2:(length(releventTime)-1)
    if candid(1,i-1) == 0 && candid(1,i) ==1 && candid(1,i+1) ==0
    candid_woOrphan(1,i) = 0; 
    else
    candid_woOrphan(1,i) = candid(1,i);     
    end  
end

% 把统计的显著性信息匹配到时间上
clusters = zeros(length(tm),1); % 记录显著性
clusterT = zeros(length(tm),1); % 记录t值
clusters(releventTime,1) = candid_woOrphan;
clusterT(releventTime,1) = Ps(1,:);
clusterTsum = sum(Ps(1,logical(candid_woOrphan))); % 总的t值

% 找到有多少个clusters，并计算每一个cluster的T值加和
tmp = zeros(10,300); % 用来记录cluster和每个cluster对应的时间点
cl = 0; % 记录cluster编号
member = 0; % 记录每一个cluster内时间点的编号
for i = 2:length(clusters)-1

        % 一个cluster的开始
        if clusters(i-1) ==0 && clusters(i) == 1 && clusters(i+1) == 1 
        cl = cl+1;
        member = member +1;
        tmp(cl,member) = i;    
        
        % 一个cluster的结束
        elseif clusters(i-1) ==1 && clusters(i) == 1 && clusters(i+1) == 0 
        member = member +1;  
        tmp(cl,member) = i;    
        member = 0;  
        
        % 一个cluster中期
        elseif clusters(i-1) ==1 && clusters(i) == 1 && clusters(i+1) == 1             
        member = member +1;  
        tmp(cl,member) = i;    
        
        else
        
        end
end

% cluster的数量
HowManyClusters = cl;
a = tmp(1:cl,:); % 提取tmp中有效的cluster
eachCluster = a(:,logical(sum(a,1)~=0)); % 提取每个cluster中有效的部分

% 计算每个cluster的t值和
dat_clusterSumT = nan(HowManyClusters,1);
for c = 1:HowManyClusters
   dat_clusterSumT(c,1) = sum(clusterT(eachCluster(c,eachCluster(c,:) ~=0)));
end

%% 进行蒙特卡洛模拟

iteration = 10000;

%% 注意：模拟会花很长时间

% 初始化模拟cluster的t值
SimclusterTvalue = nan(1,iteration);

for itr = 1:iteration
Ps = nan(2,length(releventTime));
    
    % 生成假数据，实际是随机生成decoding结果的labels

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
      
     % 对假的结果进行时间上的平滑，同真结果的平滑步骤一样
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
     % 进行t-test    
    for i = 1:length(releventTime) 
        tp = releventTime(i);
       
        [H,P,CI,STATS] =  ttest(smtFake(:,i),0.0625, 'tail','right' );

        Ps(1,i) = STATS.tstat;
        Ps(2,i) = P;
    end
    
    % 找到显著性点
    candid = Ps(2,:) <= .05;
    candid_woOrphan = zeros(1,length(candid));
    candid_woOrphan(1,1) = candid(1,1);
    candid_woOrphan(1,length(candid)) = candid(1,length(candid));
    % 去除单独的时间信息
    for i = 2:(length(releventTime)-1)

        if candid(1,i-1) == 0 && candid(1,i) ==1 && candid(1,i+1) ==0
        candid_woOrphan(1,i) = 0; 
        else
        candid_woOrphan(1,i) = candid(1,i);     
        end

    end

    % 把统计的显著性信息匹配到时间上
    clusters = zeros(length(tm),1);
    clusterT = zeros(length(tm),1);
    clusters(releventTime,1) = candid_woOrphan;
    clusterT(releventTime,1) = Ps(1,:);

    % 找到有多少个clusters，并计算每一个cluster的T值加和
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

    % 计算每个cluster的t值和
    sim_clusterSumT = zeros(HowManyClusters,1);
    for c = 1:HowManyClusters
       sim_clusterSumT(c,1) = sum(clusterT(sim_eachCluster(c,sim_eachCluster(c,:) ~=0)));
    end

    % 找到绝对值最大的T值
    record = abs(sim_clusterSumT) == max(abs(sim_clusterSumT));
    SimclusterTvalue(1,itr) = sim_clusterSumT(record);
    
    % 如果没有显著cluster，T值设为0
    else
    SimclusterTvalue(1,itr) = 0;    
    end
        
end % 模拟结束

% 对10000次迭代的T值排序
sortedTvalues = sort(SimclusterTvalue,2);

%% 找到95%的临界t
cutOff = iteration - iteration * 0.05;
critT = sortedTvalues(cutOff);
% 获取真实decoding的cluster t值大于
sigCluster = dat_clusterSumT > critT;


%% 画图

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
