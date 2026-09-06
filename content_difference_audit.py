import json, os, re, hashlib
from collections import Counter

BASE = '/root/autodl-tmp/CD-C3DA-runs'
RUNS = {
 'G0': os.path.join(BASE,'chat_G0_downstream_audit_20260905_rerun2/laptop14_to_rest15'),
 'G3': os.path.join(BASE,'chat_G3_downstream_audit_20260905_rerun2/laptop14_to_rest15'),
}
PSEUDO = {
 'G0': os.path.join(BASE,'chat_l14_r15_G0_16x2_20260904_run2_full_adapter/pseudo_variants/hp1_complete2_dist5_w025/target_pseudo_high_precision.jsonl'),
 'G3': os.path.join(BASE,'chat_l14_r15_G3_16x2_20260904_run2_full_adapter/pseudo_variants/hp1_complete2_dist5_w025/target_pseudo_high_precision.jsonl'),
}
GOLD = os.path.join(RUNS['G0'],'target_train_gold_analysis.jsonl')
PAT = re.compile(r'<(pos|neg|neu)>\s*(.*?)\s*<opinion>\s*(.*?)(?=\s*;\s*<(?:pos|neg|neu)>|$)', re.I)
def trips(label):
    return [(a.lower(),o.strip().lower(),s.lower()) for s,a,o in PAT.findall(label or '')]
def load(path): return [json.loads(x) for x in open(path, encoding='utf-8')]
def score(pred,gold):
    tp=len(pred&gold); fp=len(pred-gold); fn=len(gold-pred)
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    return {'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0,'tp':tp,'fp':fp,'fn':fn}
def main():
    rows={k:load(v) for k,v in PSEUDO.items()}
    gold={x['id']:set(trips(x.get('label',''))) for x in load(GOLD)}
    sets={k:set(t for x in rs for t in trips(x.get('label',''))) for k,rs in rows.items()}
    out={'overlap':{'G0_triplets':len(sets['G0']),'G3_triplets':len(sets['G3']),'shared_triplets':len(sets['G0']&sets['G3']),'G0_only_triplets':len(sets['G0']-sets['G3']),'G3_only_triplets':len(sets['G3']-sets['G0'])},'groups':{}}
    for k in ('G0','G3'):
        other='G3' if k=='G0' else 'G0'; only=sets[k]-sets[other]
        only_rows=[x for x in rows[k] if any(t in only for t in trips(x.get('label','')))]
        pred=set(); true=set(); sent=Counter(); lengths=[]; gold_counts=[]; kept=0; tp=fp=fn=0; cat={}
        for x in only_rows:
            pt=set(trips(x.get('label',''))); pred |= pt; g=set(gold.get(x.get('id'),set())); true |= g
            tp += len(pt & g); fp += len(pt-g); fn += len(g-pt)
            n=len(pt); c='single' if n==1 else ('3plus' if n>=3 else 'multi'); q=cat.setdefault(c,[0,0,0]); q[0]+=len(pt&g); q[1]+=len(pt-g); q[2]+=len(g-pt)
            sent.update(t[2] for t in pt); lengths.append(len(x.get('text','').split())); gold_counts.append(len(g)); kept += len(pt)
        def q(v):
            a,b,c=v; p=a/(a+b) if a+b else 0; r=a/(a+c) if a+c else 0; return {'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0,'tp':a,'fp':b,'fn':c}
        out['groups'][k]={'only_rows':len(only_rows),'only_triplets':kept,'single_rows':sum(len(trips(x.get('label','')))==1 for x in only_rows),'multi_rows':sum(len(trips(x.get('label','')))>=2 for x in only_rows),'3plus_rows':sum(len(trips(x.get('label','')))>=3 for x in only_rows),'sentiment':dict(sent),'unique_aspects':len({t[0] for t in pred}),'unique_opinions':len({t[1] for t in pred}),'mean_sentence_length':sum(lengths)/len(lengths) if lengths else 0,'mean_gold_triplets':sum(gold_counts)/len(gold_counts) if gold_counts else 0,'quality':q((tp,fp,fn)),'quality_by_structure':{c:q(v) for c,v in cat.items()},'matched_ids':sum(x.get('id') in gold for x in only_rows)}
    for k in ('G0','G3'):
        p=os.path.join(RUNS[k],'c3da_two_channel_augmented_selected_strict_aug150_w020_label_to_text_gen.jsonl'); a=load(p); out.setdefault('augmentation',{})[k]={'rows':len(a),'triplets':sum(len(trips(x.get('label',''))) for x in a),'aspect_rows':sum(x.get('augmentation')=='masked_aspect_channel' for x in a),'opinion_rows':sum(x.get('augmentation')=='masked_opinion_sentiment_channel' for x in a)}
        parent={x['id']:x for x in rows[k]}; ev={'mapping_complete':0,'mapping_missing':0,'edited_total':0,'edited_valid':0,'untouched_total':0,'untouched_preserved':0,'count_preserved':0,'count_decreased':0,'count_increased':0,'unplanned_triplets':0,'unplanned_rows':0,'by_channel':{}}
        descendants=Counter(); parent_quality=Counter(); parent_aug=Counter(); parent_seen={}; struct={}
        for x in a:
            b=parent.get(x.get('base_id'))
            if not b: ev['mapping_missing']+=1; continue
            ev['mapping_complete']+=1; pt=set(trips(b.get('label',''))); actual=set(trips(x.get('label','')))
            old=x.get('old_triplet') or []; new=x.get('new_triplet') or []
            oldt=(str(old[0]).lower(),str(old[1]).lower(),str(old[2]).lower()) if len(old)>=3 else None
            newt=(str(new[0]).lower(),str(new[1]).lower(),str(new[2]).lower()) if len(new)>=3 else None
            untouched=pt-{oldt} if oldt else pt; expected=untouched|({newt} if newt else set())
            sc='single_parent' if len(pt)==1 else ('3plus_parent' if len(pt)>=3 else 'multi_parent'); sv=struct.setdefault(sc,{'rows':0,'untouched_total':0,'untouched_preserved':0,'count_preserved':0,'count_decreased':0,'count_increased':0,'unplanned_rows':0,'unplanned_triplets':0}); sv['rows']+=1; sv['untouched_total']+=len(untouched); sv['untouched_preserved']+=len(untouched&actual)
            ev['edited_total']+=1; ev['edited_valid']+=int(newt in actual if newt else False); ev['untouched_total']+=len(untouched); ev['untouched_preserved']+=len(untouched&actual)
            diff=len(actual)-len(pt); ev['count_preserved']+=int(diff==0); ev['count_decreased']+=int(diff<0); ev['count_increased']+=int(diff>0)
            un=actual-expected; ev['unplanned_triplets']+=len(un); ev['unplanned_rows']+=int(bool(un)); sv['count_preserved']+=int(diff==0); sv['count_decreased']+=int(diff<0); sv['count_increased']+=int(diff>0); sv['unplanned_rows']+=int(bool(un)); sv['unplanned_triplets']+=len(un)
            ch='aspect' if x.get('augmentation')=='masked_aspect_channel' else 'opinion'; q=ev['by_channel'].setdefault(ch,{'rows':0,'edited_valid':0,'untouched_total':0,'untouched_preserved':0,'count_preserved':0,'count_decreased':0,'count_increased':0,'unplanned_triplets':0,'unplanned_rows':0}); q['rows']+=1; q['edited_valid']+=int(newt in actual if newt else False); q['untouched_total']+=len(untouched); q['untouched_preserved']+=len(untouched&actual); q['count_preserved']+=int(diff==0); q['count_decreased']+=int(diff<0); q['count_increased']+=int(diff>0); q['unplanned_triplets']+=len(un); q['unplanned_rows']+=int(bool(un))
            pred=set(trips(b.get('label',''))); g=set(gold.get(b.get('id'),set())); t=len(pred&g); f=len(pred-g); n=len(g-pred); quality='correct' if f==0 and n==0 else ('incorrect' if t==0 else 'partial'); parent_seen[b.get('id')]=quality; parent_aug[quality]+=1; descendants[quality]+=1
        for q in parent_seen.values(): parent_quality[q]+=1
        for sv in struct.values(): sv['untouched_retention_rate']=sv['untouched_preserved']/sv['untouched_total'] if sv['untouched_total'] else None; sv['count_preservation_rate']=sv['count_preserved']/sv['rows'] if sv['rows'] else None; sv['unplanned_rate']=sv['unplanned_rows']/sv['rows'] if sv['rows'] else None
        ev['edited_validity_rate']=ev['edited_valid']/ev['edited_total'] if ev['edited_total'] else None; ev['untouched_retention_rate']=ev['untouched_preserved']/ev['untouched_total'] if ev['untouched_total'] else None; ev['count_preservation_rate']=ev['count_preserved']/ev['edited_total'] if ev['edited_total'] else None; ev['unplanned_rate']=ev['unplanned_rows']/ev['edited_total'] if ev['edited_total'] else None; ev['parent_quality_rows']=dict(parent_quality); ev['descendant_rows_by_parent_quality']=dict(descendants); ev['mean_descendants_by_quality']={q:descendants[q]/parent_quality[q] for q in parent_quality}; ev['by_parent_structure']=struct; out.setdefault('augmentation_audit',{})[k]=ev
        f=os.path.join(RUNS[k],'final_train_strict_aug150_w020_label_to_text_gen_complete_multi2_w025_pw075.jsonl'); z=load(f); out.setdefault('final_train',{})[k]={'rows':len(z),'triplets':sum(len(trips(x.get('label',''))) for x in z),'multi_rows':sum(len(trips(x.get('label','')))>=2 for x in z),'density':sum(len(trips(x.get('label',''))) for x in z)/len(z)}
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
