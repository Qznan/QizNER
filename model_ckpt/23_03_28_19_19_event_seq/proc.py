import datautils

exms = datautils.NerExample.load_from_jsonl('test_6_3444_exm_lst.jsonl')
# datautils.NerExample.gen_html(exms, 'test.html')

datautils.NerExample.eval(exms)

for exm in exms:
    exm.aggre_ent_type(ent2ent_map={
        '冻结开始日期':'日期',
        '质押开始日期':'日期',
        '增持开始日期':'日期',
    }, other_as=None)
datautils.NerExample.eval(exms)

for exm in exms:
    exm.aggre_ent_type(ent2ent_map={}, other_as='实体')

datautils.NerExample.eval(exms)