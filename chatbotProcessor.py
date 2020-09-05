from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer

chatbot = ChatBot('Vipin',logic_adapters=['chatterbot.logic.BestMatch'])

conversation = [
    "Hi",
    "How Can I help you?",
    "Menu Items",
    "Combo Menu,Pizza Menu, Soft Drinks Menu,Others",
    "Combo Menu",
    "Combo-A(Pizza-A,Any Soft Drinks,Pasta) cost-600,Combo-B(Pizza-B,Any Soft Drinks,French Fries/Smilies) cost-580"
    "Pizza Menu",
    "Pizza-A cost-580,Pizza-B cost-550,Pizza-C cost-500",
    "Soft Drinks Menu",
    "Pepsi cost-28,Coke cost-28,7UP cost-28,Miranda cost-28",
    "Others",
    "Pasta cost-45, French Fries cost-40, Smilies cost-40, Garlic Bread cost-35",
    "Offer",
    "Buy any two pizza and get a soft drink free"
]

trainer = ListTrainer(chatbot)
trainer.train(conversation)

trainer_corpus = ChatterBotCorpusTrainer(chatbot)
trainer_corpus.train('chatterbot.corpus.english')