from model_pools.bert_final_summarizer import BertFinalSummarizer
from model_pools.bert_sentence_summarizer import BertSentenceSummarizer
from model_pools.bert_sentence_summarizer_copy import BertSentenceSummarizerCopy
from model_pools.bert_sentence_summarizer_copy_rl import BertSentenceSummarizerCopyRL
from model_pools.bert_sentence_summarizer_copy_rl_v2 import BertSentenceSummarizerCopyRLV2
from model_pools.bert_sentence_summarizer_copy_v2 import BertSentenceSummarizerCopyV2
from model_pools.bert_summarizer import BertSummarizer
from model_pools.bert_summarizer_copy import BertSummarizerCopy
from model_pools.bert_summarizer_copy_new import BertSummarizerCopyNew
from model_pools.bert_summarizer_dec import BertSummarizerDec
from model_pools.bert_summarizer_dec_draft import BertSummarizerDecDraft
from model_pools.bert_summarizer_dec_v2 import BertSummarizerDecV2
from model_pools.bert_summarizer_dec_v3 import BertSummarizerDecV3
from model_pools.bert_summarizer_dec_v4 import BertSummarizerDecV4
from model_pools.bert_summarizer_dec_v4_1 import BertSummarizerDecV4V1
from model_pools.bert_summarizer_dec_v4_2 import BertSummarizerDecV4V2
from model_pools.bert_summarizer_dec_v4_3 import BertSummarizerDecV4V3
from model_pools.bert_summarizer_pre_train import BertSummarizerPreTrain
from model_pools.bert_two_stage_summarizer import BertTwoStageSummarizer
from model_pools.bert_two_stage_summarizer_v2 import BertTwoStageSummarizerV2
from model_pools.baseline_copy_same_vocab import BertSummarizerCopySameVocab
from model_pools.baseline_copy_same_vocab_multi_gpu import BertSummarizerCopySameVocabMultiGPU
from model_pools.topic_baseline_en_concat import TopicSummarizerEnConcat
from model_pools.topic_baseline_en_concat_memory import TopicSummarizerEnConcatMem
from model_pools.topic_baseline_en_concat_2memory import TopicSummarizerEnConcat2Mem
from model_pools.encoder_filter_topic_baseline import TopicEncoderFilter
from model_pools.encoder_filter_topic_decoder_mem import TopicEncoderFilterDecoderMem
from model_pools.encoder_filter_topic_decoder_mem_copy import TopicEncoderFilterDecoderMemCopy
from model_pools.encoder_filter_query_baseline import QueryEncoderFilter
from model_pools.encoder_filter_query_decoder_mem import QueryEncoderFilterDecoderMem
from model_pools.encoder_filter_query_plus_decoder_mem import QueryEncoderFilterPlusDecoderMem
from model_pools.encoder_filter_query_plus_decoder_mem_copy import QueryEncoderFilterPlusDecoderMemCopy
from model_pools.encoder_filter_query_decoder_mem_copy import QueryEncoderFilterDecoderMemCopy
from model_pools.encoder_filter_query_plus import QueryEncoderFilterPlus
from model_pools.encoder_filter_query_stack import QueryEncoderFilterStack
from model_pools.encoder_filter_query_stack_de import QueryEncoderFilterStackDecoderMem
from model_pools.encoder_filter_query_parallel import QueryEncoderFilterParallel
from model_pools.encoder_filter_query_parallel_de import QueryEncoderFilterParallelDecoderMem

model_pools = {
    'summarize_bert_baseline': BertSummarizer,
    'summarize_bert_baseline_copy': BertSummarizerCopy,
    'bert_two_stage_summarizer': BertTwoStageSummarizer,
    'bert_two_stage_summarizer_v2': BertTwoStageSummarizerV2,
    'bert_final_summarizer': BertFinalSummarizer,
    'bert_sentence_summarizer': BertSentenceSummarizer,
    'bert_sentence_summarizer_copy': BertSentenceSummarizerCopy,
    'bert_sentence_summarizer_copy_v2': BertSentenceSummarizerCopyV2,
    'bert_sentence_summarizer_copy_rl': BertSentenceSummarizerCopyRL,
    'bert_sentence_summarizer_copy_rl_v2': BertSentenceSummarizerCopyRLV2,
    'bert_summarizer_dec': BertSummarizerDec,
    'bert_summarizer_dec_v2': BertSummarizerDecV2,
    'bert_summarizer_dec_v3': BertSummarizerDecV3,
    'bert_summarizer_dec_v4': BertSummarizerDecV4,
    'bert_summarizer_dec_v4_1': BertSummarizerDecV4V1,
    'bert_summarizer_dec_v4_2': BertSummarizerDecV4V2,
    'bert_summarizer_dec_v4_3': BertSummarizerDecV4V3,
    'bert_summarizer_dec_draft': BertSummarizerDecDraft,
    'bert_summarizer_pre_train': BertSummarizerPreTrain,
    'bert_summarizer_copy_new': BertSummarizerCopyNew,
    'baseline_copy_same_vocab': BertSummarizerCopySameVocab,
    'baseline_copy_same_vocab_multi_gpu': BertSummarizerCopySameVocabMultiGPU,
    'topic_baseline_en_concat': TopicSummarizerEnConcat,
    'topic_baseline_en_concat_mem': TopicSummarizerEnConcatMem,
    'topic_baseline_en_concat_2mem':TopicSummarizerEnConcat2Mem,
    'encoder_filter_topic_baseline':TopicEncoderFilter,
    'encoder_filter_topic_decoder_mem':TopicEncoderFilterDecoderMem,
    #'encoder_filter_topic_plus_decoder_mem':TopicEncoderFilterPlusDecoderMem,
    'encoder_filter_topic_decoder_mem_copy':TopicEncoderFilterDecoderMemCopy,
    #'encoder_filter_topic_plus_decoder_mem_copy':TopicEncoderFilterPlusDecoderMemCopy,
    'encoder_filter_query_baseline':QueryEncoderFilter,
    'encoder_filter_query_plus':QueryEncoderFilterPlus,
    'encoder_filter_query_stack':QueryEncoderFilterStack,
    'encoder_filter_query_parallel':QueryEncoderFilterParallel,
    'encoder_filter_query_stack_decoder_mem':QueryEncoderFilterStackDecoderMem,
    'encoder_filter_query_parallel_decoder_mem':QueryEncoderFilterParallelDecoderMem,
    'encoder_filter_query_decoder_mem':QueryEncoderFilterDecoderMem,
    'encoder_filter_query_plus_decoder_mem':QueryEncoderFilterPlusDecoderMem,
    'encoder_filter_query_decoder_mem_copy':QueryEncoderFilterDecoderMemCopy,
    'encoder_filter_query_plus_decoder_mem_copy':QueryEncoderFilterPlusDecoderMemCopy
}
