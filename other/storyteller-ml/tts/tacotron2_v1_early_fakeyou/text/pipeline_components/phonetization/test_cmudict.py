from text.pipeline_components.phonetization.cmudict import CMUDict
from text.pipeline_components.phonetization.cmudict import DEFAULT_CMUDICT_PATH


class TestCmuDict:

    def test_lookup_success(self):
        cmudict = CMUDict(DEFAULT_CMUDICT_PATH)
        arpabet_sequence = cmudict.lookup('word')
        assert arpabet_sequence == ['W ER1 D'] # TODO: Split the phonemes!

    def test_lookup_failure(self):
        cmudict = CMUDict(DEFAULT_CMUDICT_PATH)
        arpabet_sequence = cmudict.lookup('fakeword')
        assert arpabet_sequence == None
