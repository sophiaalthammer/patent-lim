import pytest
from ling_ana import ling_ana_wiki


@pytest.fixture
def sentences():
    return [' Robert Boulter is an English film , television and theatre actor .',
            'He had a guest @-@ starring role on the television series The Bill in 2000 .',
            'This was followed by a starring role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre .',
            'He had a guest role in the television series Judge John Deed in 2002 .',
            'In 2004 Boulter landed a role as " Craig " in the episode " Teddy \'s Story " of the television series The Long Firm ; he starred alongside actors Mark Strong and Derek Jacobi .',
            'He was cast in the 2005 theatre productions of the Philip Ridley play Mercury Fur , which was performed at the Drum Theatre in Plymouth and the Menier Chocolate Factory in London .',
            'He was directed by John Tiffany and starred alongside Ben Whishaw , Shane Zaza , Harry Kent , Fraser Ayres , Sophie Stanton and Dominic Hall .',
            ' In 2006 , Boulter starred alongside Whishaw in the play Citizenship written by Mark Ravenhill .',
            'He appeared on a 2006 episode of the television series , Doctors , followed by a role in the 2007 theatre production of How to Curse directed by Josie Rourke .',
            'How to Curse was performed at Bush Theatre in the London Borough of Hammersmith and Fulham .',
            'Boulter starred in two films in 2008 , Daylight Robbery by filmmaker Paris Leonti , and Donkey Punch directed by Olly Blackburn .',
            'In May 2008 , Boulter made a guest appearance on a two @-@ part episode arc of the television series Waking the Dead , followed by an appearance on the television series Survivors in November 2008 .',
            'He had a recurring role in ten episodes of the television series Casualty in 2010 , as " Kieron Fletcher " .',
            'Boulter starred in the 2011 film Mercenaries directed by Paris Leonti .',
            ' In 2000 Boulter had a guest @-@ starring role on the television series The Bill ; he portrayed " Scott Parry " in the episode , " In Safe Hands " .',
            'Boulter starred as " Scott " in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre .',
            'A review of Boulter \'s performance in The Independent on Sunday described him as " horribly menacing " in the role , and he received critical reviews in The Herald , and Evening Standard .',
            'He appeared in the television series Judge John Deed in 2002 as " Addem Armitage " in the episode " Political Expediency " , and had a role as a different character " Toby Steele " on The Bill .',
            ' He had a recurring role in 2003 on two episodes of The Bill , as character " Connor Price " .',
            'In 2004 Boulter landed a role as " Craig " in the episode " Teddy \'s Story " of the television series The Long Firm ; he starred alongside actors Mark Strong and Derek Jacobi .',
            'Boulter starred as " Darren " , in the 2005 theatre productions of the Philip Ridley play Mercury Fur .',
            'It was performed at the Drum Theatre in Plymouth , and the Menier Chocolate Factory in London .',
            'He was directed by John Tiffany and starred alongside Ben Whishaw , Shane Zaza , Harry Kent , Fraser Ayres , Sophie Stanton and Dominic Hall .',
            'Boulter received a favorable review in The Daily Telegraph : " The acting is shatteringly intense , with wired performances from Ben Whishaw ( now unrecognisable from his performance as Trevor Nunn \'s Hamlet ) , Robert Boulter , Shane Zaza and Fraser Ayres . "',
            'The Guardian noted , " Ben Whishaw and Robert Boulter offer tenderness amid the savagery . "',
            ' In 2006 Boulter starred in the play Citizenship written by Mark Ravenhill .',
            'The play was part of a series which featured different playwrights , titled Burn / Chatroom / Citizenship .',
            'In a 2006 interview , fellow actor Ben Whishaw identified Boulter as one of his favorite co @-@ stars : " I loved working with a guy called Robert Boulter , who was in the triple bill of Burn , Chatroom and Citizenship at the National .',
            'He played my brother in Mercury Fur . "',
            'He portrayed " Jason Tyler " on the 2006 episode of the television series , Doctors , titled " Something I Ate " .',
            'Boulter starred as " William " in the 2007 production of How to Curse directed by Josie Rourke .',
            'How to Curse was performed at Bush Theatre in the London Borough of Hammersmith and Fulham .',
            'In a review of the production for The Daily Telegraph , theatre critic Charles Spencer noted , " Robert Boulter brings a touching vulnerability to the stage as William . "',
            ' Boulter starred in two films in 2008 , Daylight Robbery by filmmaker Paris Leonti , and Donkey Punch directed by Olly Blackburn .',
            'Boulter portrayed a character named " Sean " in Donkey Punch , who tags along with character " Josh " as the " quiet brother ... who hits it off with Tammi " .',
            'Boulter guest starred on a two @-@ part episode arc " Wounds " in May 2008 of the television series Waking the Dead as character " Jimmy Dearden " .',
            'He appeared on the television series Survivors as " Neil " in November 2008 .',
            'He had a recurring role in ten episodes of the television series Casualty in 2010 , as " Kieron Fletcher " .',
            'He portrayed an emergency physician applying for a medical fellowship .',
            'He commented on the inherent difficulties in portraying a physician on television : " Playing a doctor is a strange experience .',
            'Pretending you know what you \'re talking about when you don \'t is very bizarre but there are advisers on set who are fantastic at taking you through procedures and giving you the confidence to stand there and look like you know what you \'re doing . "',
            'Boulter starred in the 2011 film Mercenaries directed by Paris Leonti .',
            ' Du Fu ( Wade – Giles : Tu Fu ; Chinese : 杜甫 ; 712 – 770 ) was a prominent Chinese poet of the Tang dynasty .',
            'Along with Li Bai ( Li Po ) , he is frequently called the greatest of the Chinese poets .',
            'His greatest ambition was to serve his country as a successful civil servant , but he proved unable to make the necessary accommodations .',
            'His life , like the whole country , was devastated by the An Lushan Rebellion of 755 , and his last 15 years were a time of almost constant unrest .',
            ' Although initially he was little @-@ known to other writers , his works came to be hugely influential in both Chinese and Japanese literary culture .',
            'Of his poetic writing , nearly fifteen hundred poems have been preserved over the ages .',
            'He has been called the " Poet @-@ Historian " and the " Poet @-@ Sage " by Chinese critics , while the range of his work has allowed him to be introduced to Western readers as " the Chinese Virgil , Horace , Ovid , Shakespeare , Milton , Burns , Wordsworth , Béranger , Hugo or Baudelaire " .',
            ' Traditional Chinese literary criticism emphasized the life of the author when interpreting a work , a practice which Burton Watson attributes to " the close links that traditional Chinese thought posits between art and morality " .',
            "Since many of Du Fu 's poems feature morality and history , this practice is particularly important .",
            'Another reason , identified by the Chinese historian William Hung , is that Chinese poems are typically concise , omitting context that might be relevant , but which an informed contemporary could be assumed to know .',
            'For modern Western readers , " The less accurately we know the time , the place and the circumstances in the background , the more liable we are to imagine it incorrectly , and the result will be that we either misunderstand the poem or fail to understand it altogether " .',
            'Stephen Owen suggests a third factor particular to Du Fu , arguing that the variety of the poet \'s work required consideration of his whole life , rather than the " reductive " categorizations used for more limited poets .',
            " Most of what is known of Du Fu 's life comes from his poems .",
            'His paternal grandfather was Du Shenyan , a noted politician and poet during the reign of Empress Wu .',
            'Du Fu was born in 712 ; the exact birthplace is unknown , except that it was near Luoyang , Henan province ( Gong county is a favourite candidate ) .',
            "In later life , he considered himself to belong to the capital city of Chang 'an , ancestral hometown of the Du family .",
            " Du Fu 's mother died shortly after he was born , and he was partially raised by his aunt .",
            'He had an elder brother , who died young .',
            'He also had three half brothers and one half sister , to whom he frequently refers in his poems , although he never mentions his stepmother .',
            ' The son of a minor scholar @-@ official , his youth was spent on the standard education of a future civil servant : study and memorisation of the Confucian classics of philosophy , history and poetry .',
            'He later claimed to have produced creditable poems by his early teens , but these have been lost .',
            ' In the early 730s , he travelled in the Jiangsu / Zhejiang area ; his earliest surviving poem , describing a poetry contest , is thought to date from the end of this period , around 735 .',
            "In that year , he took the civil service exam , likely in Chang 'an .",
            'He failed , to his surprise and that of centuries of later critics .',
            'Hung concludes that he probably failed because his prose style at the time was too dense and obscure , while Chou suggests his failure to cultivate connections in the capital may have been to blame .',
            'After this failure , he went back to traveling , this time around Shandong and Hebei .',
            ' His father died around 740 .',
            "Du Fu would have been allowed to enter the civil service because of his father 's rank , but he is thought to have given up the privilege in favour of one of his half brothers .",
            'He spent the next four years living in the Luoyang area , fulfilling his duties in domestic affairs .',
            ' In the autumn of 744 , he met Li Bai ( Li Po ) for the first time , and the two poets formed a friendship .',
            'David Young describes this as " the most significant formative element in Du Fu \'s artistic development " because it gave him a living example of the reclusive poet @-@ scholar life to which he was attracted after his failure in the civil service exam .',
            'The relationship was somewhat one @-@ sided , however .',
            'Du Fu was by some years the younger , while Li Bai was already a poetic star .',
            'We have twelve poems to or about Li Bai from the younger poet , but only one in the other direction .',
            'They met again only once , in 745 .',
            ' In 746 , he moved to the capital in an attempt to resurrect his official career .',
            'He took the civil service exam a second time during the following year , but all the candidates were failed by the prime minister ( apparently in order to prevent the emergence of possible rivals ) .',
            'He never again attempted the examinations , instead petitioning the emperor directly in 751 , 754 and probably again in 755 .',
            'He married around 752 , and by 757 the couple had had five children — three sons and two daughters — but one of the sons died in infancy in 755 .',
            'From 754 he began to have lung problems ( probably asthma ) , the first of a series of ailments which dogged him for the rest of his life .',
            'It was in that year that Du Fu was forced to move his family due to the turmoil of a famine brought about by massive floods in the region .',
            " In 755 , he received an appointment as Registrar of the Right Commandant 's office of the Crown Prince 's Palace .",
            'Although this was a minor post , in normal times it would have been at least the start of an official career .',
            'Even before he had begun work , however , the position was swept away by events .',
            ' The An Lushan Rebellion began in December 755 , and was not completely suppressed for almost eight years .',
            'It caused enormous disruption to Chinese society : the census of 754 recorded 52 @.',
            '@ 9 million people , but ten years later , the census counted just 16 @.',
            '@ 9 million , the remainder having been displaced or killed .',
            'During this time , Du Fu led a largely itinerant life unsettled by wars , associated famines and imperial displeasure .',
            'This period of unhappiness was the making of Du Fu as a poet : Even Shan Chou has written that , " What he saw around him — the lives of his family , neighbors , and strangers – what he heard , and what he hoped for or feared from the progress of various campaigns — these became the enduring themes of his poetry " .',
            'Even when he learned of the death of his youngest child , he turned to the suffering of others in his poetry instead of dwelling upon his own misfortunes .',
            'Du Fu wrote :',
            ' Brooding on what I have lived through , if even I know such suffering , the common man must surely be rattled by the winds .',
            ' In 756 , Emperor Xuanzong was forced to flee the capital and abdicate .',
            "Du Fu , who had been away from the city , took his family to a place of safety and attempted to join the court of the new emperor ( Suzong ) , but he was captured by the rebels and taken to Chang 'an .",
            'In the autumn , his youngest son , Du Zongwu ( Baby Bear ) , was born .',
            'Around this time Du Fu is thought to have contracted malaria .',
            " He escaped from Chang 'an the following year , and was appointed Reminder when he rejoined the court in May 757 ."]


@pytest.fixture
def sentences2():
    return [' Robert Boulter is an English film , television and theatre actor .',
            'He had a guest @-@ starring role on the television series The Bill in 2000 .']


def test_get_hyphen_exp(sentences):
    assert ling_ana_wiki.get_hyphen_exp_wiki(sentences) == ['guest-starring', 'two-part', 'guest-starring', 'co-stars',
                                                            'two-part', 'little-known', 'Poet-Historian', 'Poet-Sage',
                                                            'scholar-official', 'poet-scholar', 'one-sided']


def test_count_words_wiki():
    assert ling_ana_wiki.count_words_wiki(' = Robert Boulter = ') == 2


def test_counts_words_wiki2():
    assert ling_ana_wiki.count_words_wiki(
        'He had a guest @-@ starring role on the television series The Bill in 2000 .') == 14


def test_replace_hyphen_sentences(sentences2):
    assert ling_ana_wiki.replace_hyphen_wiki(sentences2) == [
        ' Robert Boulter is an English film , television and theatre actor .',
        'He had a guest - starring role on the television series The Bill in 2000 .']
