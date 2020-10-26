rng('default');
emb = fastTextWordEmbedding;  % incarcam o baza de date *antrenata deja* cu cuvinte "embedded" cu vectori numerici.

data = readLexicon; % aici avem o alta baza de date cu cuvinte ( vreo 6000 cuvinte) ce au label pozitiv/negativ

idx = ~isVocabularyWord(emb,data.Word); % stergem cuvintele din baza de date care nu se regasesc in baza de date "emb"
data(idx,:) = [];    %scriem data.Word pt ca data e o structura cu Word si Label

numWords = size(data,1);
cvp = cvpartition(numWords,'HoldOut',0.1); % 10% din cuvintele ramase le pastram pentru test si nu le vom folosi la antrenare
dataTrain = data(training(cvp),:);
dataTest = data(test(cvp),:);

wordsTrain = dataTrain.Word; %folosim doar cuvintele de train
XTrain = word2vec(emb,wordsTrain);  % aici transformam cuvintele in vectori numerici, acest fapt se bazeaza pe acel emb care este baza de date a cuvintelor embedded cu vectori numerici
YTrain = dataTrain.Label; % Ytrain contine labelurile ( Fiind o clasificare binara, este o invatare de tip supervizaata. Astfel, la antrenare avem nevoie si de etichete)

mdl = fitcsvm(XTrain,YTrain); % Antrenam modelul folosind svm

% Acum vom vedea rezultatele modelului asupra datelor de test ( vedem cum
% prezice)

wordsTest = dataTest.Word;
XTest = word2vec(emb,wordsTest);
YTest = dataTest.Label;

[YPred,scores] = predict(mdl,XTest); %prezicem labelurile pentru cuvintele ce nu au fost folosite la antrenare.
%verificam rezultatele cu ajutorul matricii de confuzie

figure
confusionchart(YTest,YPred);

figure % aici vedem rezultatele pe wordcloud
subplot(1,2,1)
idx = YPred == "Positive";
wordcloud(wordsTest(idx),scores(idx,1));
title("Cuvinte cu sentiment pozitiv")

subplot(1,2,2)
wordcloud(wordsTest(~idx),scores(~idx,2));
title("Cuvinte cu sentiment negativ")

%Pana acum ne-am antrenat un model si am testat pe niste cuvinte cu
%labeluri.
%Pentru a vedea ce fel de sentiment ne ofera un text vom lua fiecare cuvant
%din el si vom calcula "scorul lui". Daca este scor pozitiv, cuvantul este
%pozitiv, altfel este negativ. Sentimentul oferit de text va fi dat de
%media scorurilor fiecarui cuvant din text. Principiul este acelasi.

filename = "final_ordonat.txt";
tbl = readtable(filename,'TextType','string', 'Delimiter', '\n'); % citim propozitiile, vom avea un vector de stringuri.
textData = tbl.TextData;

documents = preprocessText(textData); %aici preprocesam textul(scoatem semnele de punctuatie, stopwords, lower etc)

idx = ~isVocabularyWord(emb,documents.Vocabulary); %stergem cuvintele care nu apar in vocabularul emb
documents = removeWords(documents,idx);

%for i = 1:numel(documents)
 %   words = string(documents(i));
  %  vec = word2vec(emb,words);
  %  [~,scores] = predict(mdl,vec);
   % sentimentScore(i) = mean(scores(:,1));
%end

for i = 2:50001
    words = string(documents(i));
    vec = word2vec(emb, words);
    [~, scores] = predict(mdl, vec);
    sentimentScore(i) = mean(scores(:,1));
end

corect_poz = 0;
for i = 2:25001
    if sentimentScore(i) > 0
        corect_poz = corect_poz + 1;
    end
end

corect_neg = 0;
for i = 25002:50001
    if sentimentScore(i) < 0
        corect_neg = corect_neg + 1;
    end
end

accuracy_poz = corect_poz / 25000
accuracy_neg = corect_neg / 25000
acurracy_total = (accuracy_poz + accuracy_neg) / 2

