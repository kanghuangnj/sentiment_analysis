import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw
text = 'i need printing set up on a new computer in the office'
aug = naw.SynonymAug(aug_src='ppdb', model_path='../data/ppdb-2.0-tldr')
augmented_text = aug.augment(text)
