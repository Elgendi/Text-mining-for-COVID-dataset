clear;clc;close all;
rng('default') % setting random number seed for reproducibility
load comm.mat; % load database

AllAbstract=t.Abstract;
AllTitle=t.Title;

tda = preprocessText(AllAbstract); %tokenize, removeStopWords, erasePunctuation,...
numDocuments = numel(tda); %returns number of elements in tda

n_array=[100,200,300,400]; %select population sizes (paper uses [100,200,300,400])


LookingFor=["influenza virus"]; %define search term 

holdout=0.1; % set factor for size of validation set 

%% fit all LDA-models and get their rankings

for j = 1:numel(n_array) %iterate over populations
n=n_array(j)    

documentsTrain = tda(1:n); %select n training documents
documentsValidation = tda((numDocuments-n*holdout-1):end); % 10% of n documents are selected for validation

bag = bagOfNgrams(documentsTrain); %creates bigrams
validationData = bagOfNgrams(documentsValidation); %same for validation



numTopicsRange=[1:n]; %defines number of topics k

for i = 1:numel(numTopicsRange)  % fit a model for each number of topics
    numTopics = numTopicsRange(i);
    
    mdl = fitlda(bag,numTopics, ... % model fit
        'Solver','savb', ...
        'Verbose',0);
    
    [~,validationPerplexity(i)] = logp(mdl,documentsValidation); %calculate validation perplexity with logp
    timeElapsed(i) = mdl.FitInfo.History.TimeSinceStart(end); % measure time to fit
    

    scoresDoc = mdl.DocumentTopicProbabilities; 
    scoresWord = mdl.TopicWordProbabilities;
    transmissionid = find(contains(mdl.Vocabulary, LookingFor));
    [maxprobWord,idxTopicTrans] = max(scoresWord(transmissionid, :));
    [p,idx]=max(maxprobWord);   % find topic (idx) that corresponds most to "influenza virus"

    [~, idxTopicDoc]= sort(scoresDoc(:, idx), 'descend'); %idxTopicDoc is the sorted doc index  
    BestMatch(i,:)= idxTopicDoc(1:5); %save top 5 papers of the model
   
    % visualization for 1 example
    if numTopics==4 && n==100
        figure
        for topicIdx = 1:numTopics
            subplot(2,2,topicIdx)
            wordcloud(mdl,topicIdx);
            title("Topic: " + topicIdx)
        end
        idx % print matching topic
        idxTopicDoc(1) % print best paper
        exportgraphics(gcf,'wordcloud_example.pdf','ContentType','vector') %save graphic (figure 3 of paper)
    end
    
end

MatchMat=[numTopicsRange', BestMatch]; %create matrix with k in first column and the best match in the 2nd.

% majority vote:
list=[1:n]; 
distribution=hist(MatchMat(:,2:6),list); %count paper frequency
[dist_occ, dist_idx]=sort(distribution, 'descend'); 

sort_dist10= dist_idx(1:10); %get top 10 ranking of current population
save(['sort',num2str(n),'.mat'], 'sort_dist10'); %save ranking


end

%%  use all rankings for overall majority vote

for i=1:numel(n_array) %import all rankings
n=n_array(i);
filename=['sort',num2str(n),'.mat'];
load(filename); 
sort10(:,i)=sort_dist10;
end

ranking_table=[n_array;sort10]; %combine to 1 table

save('ranking_table.mat', 'ranking_table'); %save ranking table (Table 1 of paper)

n=n_array(end);
list=[1:n];
distribution=hist(ranking_table(2:end,:),list); 
dist_comp=sum(distribution,2); %calculate frequency over table

app=max(dist_comp); % find maximum appearance
output=find(dist_comp==app); % find paper indeces 
output_title=AllTitle(output);

output %print output paper index
output_title %print output paper title