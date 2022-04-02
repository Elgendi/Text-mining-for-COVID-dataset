
function [documents] = preprocessText(textData)
% Convert the text data to lowercase.
textData = lower(textData);

% Tokenize the text.
documents = tokenizedDocument(textData, 'Language', 'en');

% Erase punctuation.
documents = erasePunctuation(documents);

% Remove a list of stop words.
documents = removeStopWords(documents);

% Remove words with 2 or fewer characters, and words with 15 or greater
% characters.
documents = removeShortWords(documents,2);
documents = removeLongWords(documents,15);

% Lemmatize the words.
documents = addPartOfSpeechDetails(documents);
documents = normalizeWords(documents,'Style','lemma');
end