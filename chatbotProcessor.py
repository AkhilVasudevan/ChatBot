from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer

chatbot = ChatBot('Vipin',logic_adapters=[
        {
            'import_path':'chatterbot.logic.BestMatch',
            'default_response': 'I am sorry, but I do not understand.',
            'maximum_similarity_threshold': 0.70
        }
    ])

with open('./Dataset/greetings.txt') as f:
    greetings = [line.rstrip() for line in f]
f.close()
with open('./Dataset/Menu.txt') as f:
    menus=[line.rstrip() for line in f]
f.close()

trainer = ListTrainer(chatbot)
trainer.train(greetings)
trainer.train(menus)

trainer_corpus = ChatterBotCorpusTrainer(chatbot)
trainer_corpus.train('chatterbot.corpus.english')