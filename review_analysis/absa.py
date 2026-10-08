"""속성 기반 감성분석(ABSA): 의류 리뷰에서 속성별 감성(긍정/부정/중립)을 Gemini로 추출하고,
언어·번역·실행 간 일관성과 합성 리뷰 정답 대비 정확도를 잰다. 라벨 기준은 docs/absa-label-guide.md."""
import json
import random
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import config as C
from .translators import GeminiClient

# (속성, 구분). 구분: 상품 / 서비스
ASPECT_DEFS = [
    ('재질·촉감', '상품'), ('사이즈·핏', '상품'), ('색상·디자인', '상품'), ('품질·내구성', '상품'),
    ('가격·가성비', '상품'), ('냄새', '상품'), ('세탁·관리', '상품'),
    ('배송', '서비스'), ('포장', '서비스'), ('교환·환불·CS', '서비스'),
]
ASPECTS = [a for a, _ in ASPECT_DEFS]
SERVICE = [a for a, k in ASPECT_DEFS if k == '서비스']
SENTS = ('긍정', '부정', '중립')
OUT = C.OUT_DIR / 'absa_pilot'
VERSION = 'v1'   # 프롬프트/가이드가 바뀌면 올려서 캐시를 분리한다 (v0 = 가이드 확정 전 시험)

PROMPT = (
    '다음은 의류·패션 상품 리뷰 목록입니다(영어 또는 한국어). 각 리뷰에서 아래 속성 중 리뷰가 실제로 언급한 속성만 골라 속성별 감성을 판단하세요.\n'
    f'속성 목록: {", ".join(ASPECTS)}\n'
    '규칙: 리뷰가 그 속성을 평가한 경우에만 출력하세요. 평가 표현 없이 사실만 적은 경우(예: "택배 박스에 담겨 왔다")는 언급하지 않은 것으로 보고 출력하지 마세요. '
    '감성은 "긍정"(분명한 만족), "부정"(분명한 불만), "중립"(보통/무난/그저 그렇다고 평가했거나 장단점이 같은 비중으로 섞임) 중 하나입니다. '
    '한 속성에 긍정과 부정이 섞이면 전체 뉘앙스로, 반어는 실제 뜻으로 판단하세요. 속성 목록에 없는 내용은 무시하세요.\n'
    '출력: [{"id": 번호, "aspects": [{"aspect": "속성", "sentiment": "감성"}]}] 형식의 JSON 배열만 출력하세요.\n\n입력:\n')


def _parse(got):
    res = []
    for g in got:
        d = {}
        for a in (g or []):
            if isinstance(a, dict) and a.get('aspect') in ASPECTS and a.get('sentiment') in SENTS:
                d[a['aspect']] = a['sentiment']
        res.append(d)
    return res


def extract(name, texts, out_dir=OUT):
    """texts의 속성별 감성을 Gemini로 추출. 같은 name의 저장본이 있으면 재사용한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{name}.{VERSION}.json'
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if len(saved) == len(texts):
            return saved
    got = GeminiClient(C.LLM_JUDGE_MODEL).batched(PROMPT, texts, 'review', 'aspects', desc=f'ABSA {name}')
    res = _parse(got)
    path.write_text(json.dumps(res, ensure_ascii=False), encoding='utf-8')
    return res


def compare(a, b):
    """a, b: 리뷰별 {속성: 감성}. 속성 탐지 정밀도/재현율/F1, 공통 속성의 감성 일치율, 극성 반전율."""
    tp = fp = fn = agree = flip = both = 0
    for x, y in zip(a, b):
        sx, sy = set(x), set(y)
        tp += len(sx & sy); fp += len(sy - sx); fn += len(sx - sy)
        for k in sx & sy:
            both += 1
            agree += x[k] == y[k]
            flip += {x[k], y[k]} == {'긍정', '부정'}
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return dict(탐지_정밀도=round(p, 3), 탐지_재현율=round(r, 3), 탐지_F1=round(2 * p * r / max(1e-9, p + r), 3),
                공통속성수=both, 감성일치율=round(agree / max(1, both), 3), 극성반전율=round(flip / max(1, both), 3))


def per_aspect(gold, pred):
    """속성별 탐지 정밀도/재현율/F1과 감성 정확도 (gold 기준)."""
    rows = {}
    for asp in ASPECTS:
        tp = sum(asp in g and asp in p for g, p in zip(gold, pred))
        fp = sum(asp not in g and asp in p for g, p in zip(gold, pred))
        fn = sum(asp in g and asp not in p for g, p in zip(gold, pred))
        sacc = [g[asp] == p[asp] for g, p in zip(gold, pred) if asp in g and asp in p]
        pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
        rows[asp] = dict(정답수=tp + fn, 정밀도=round(pr, 3), 재현율=round(rc, 3),
                         F1=round(2 * pr * rc / max(1e-9, pr + rc), 3), 감성정확도=round(float(np.mean(sacc)), 3) if sacc else None)
    return pd.DataFrame(rows).T


# ---------- 한글 합성 리뷰 (정답이 생성 시점에 정해짐) ----------
STYLES = ['짧은 반말체(예: "배송 빠름 근데 좀 작음")', 'ㅋㅋ/ㅠㅠ 같은 표현과 오타가 조금 섞인 말투', '정중하고 자세한 후기체',
          '불만을 비꼬는 반어 표현이 한 번 들어간 말투', '이모지 없이 간결한 두세 문장', '구매 의도를 설명하는 긴 후기(4~5문장)',
          '재구매 의사를 밝히는 말투', '사진 후기 느낌의 짧은 코멘트']

SYNTH_PROMPT = (
    '당신은 한국 온라인 쇼핑몰(쿠팡, 네이버 스마트스토어 등)의 의류 구매 후기를 쓰는 사람입니다. 아래 각 항목의 지시에 맞는 후기를 한 개씩 쓰세요.\n'
    '규칙: 지시된 속성은 모두 지시된 감성으로 드러나게 쓰고, 항목의 "forbidden"에 적힌 속성은 단어와 암시 모두 절대 언급하지 마세요 '
    '(예: 배송이 forbidden이면 택배, 도착, 배달도 쓰지 말고, 품질이 forbidden이면 튼튼함, 올 풀림, 마감도 쓰지 마세요). '
    '"중립"은 그 속성을 "보통이에요, 무난해요, 그냥 그래요"처럼 평가한 것이지 사실만 적는 것이 아닙니다. 중립 속성에 칭찬이나 불만을 섞지 마세요. '
    '"긍정/부정/중립" 같은 단어를 직접 쓰지 말고 실제 후기처럼 자연스럽게 쓰세요. 말투는 지시한 스타일을 따르세요. 상품명이나 브랜드는 쓰지 마세요.\n'
    '출력: [{"id": 번호, "review": "후기"}] 형식의 JSON 배열만 출력하세요.\n\n입력:\n')


def make_specs(n, seed=C.SEED, service_ratio=0.45):
    """합성 리뷰 지시서. 서비스 속성이 들어간 항목을 service_ratio만큼 넣어 아마존 데이터의 약점을 보완한다."""
    rng = random.Random(seed)
    specs = []
    for i in range(n):
        k = rng.choices([1, 2, 3], [0.4, 0.4, 0.2])[0]
        pool_service = rng.random() < service_ratio
        first = rng.choice(SERVICE if pool_service else ASPECTS)
        asps = [first] + rng.sample([a for a in ASPECTS if a != first], k - 1)
        specs.append(dict(id=i, style=rng.choice(STYLES),
                          aspects=[dict(aspect=a, sentiment=rng.choice(SENTS)) for a in asps],
                          forbidden=[a for a in ASPECTS if a not in asps]))
    return specs


def synth_reviews(n=200, batch=10):
    """Gemini로 한글 합성 리뷰를 만들고 정답과 함께 저장한다. (같은 모델이 쓰고 읽으므로 정확도는 낙관적이다)"""
    path = OUT / f'synth_ko.{VERSION}.json'
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if len(saved) == n:
            return saved
    specs = make_specs(n)
    gc = GeminiClient(C.LLM_JUDGE_MODEL)
    out = []
    for i in tqdm(range(0, n, batch), desc='합성 리뷰'):
        part = specs[i:i + batch]
        got = gc.call(SYNTH_PROMPT, part, 'review')
        for s in part:
            out.append(dict(s, review=str(got.get(s['id']) or '').strip()))
        time.sleep(C.GEMINI_SLEEP)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
    return out


def validate_synth(data):
    """합성 리뷰를 읽기 전용 추출로 다시 검증한다. 정답(지시서)과 추출이 완전히 같은 항목만 clean으로 분류하고,
    다른 항목은 사람이 확인할 수 있게 review_queue로 돌려준다."""
    data = [d for d in data if d['review']]
    pred = extract(f'synth_pred_{len(data)}', [d['review'] for d in data])
    gold = [{x['aspect']: x['sentiment'] for x in d['aspects']} for d in data]
    clean = [i for i, (g, p) in enumerate(zip(gold, pred)) if g == p]
    queue = [dict(idx=i, review=data[i]['review'], 지시=json.dumps(gold[i], ensure_ascii=False),
                  추출=json.dumps(pred[i], ensure_ascii=False)) for i in range(len(data)) if i not in set(clean)]
    return data, gold, pred, clean, queue


def score_eval_csv(path):
    """사람이 작성한 평가셋 CSV(긴 형식: review_id,text,aspect,sentiment,annotator,note)를 Gemini 추출과 비교한다.
    annotator가 둘 이상이면 첫 번째를 정답으로 쓰고 둘 사이 일치도도 보고한다. 예시 행(annotator=예시)은 제외한다."""
    df = pd.read_csv(path, encoding='utf-8-sig')
    df = df[df['annotator'].astype(str) != '예시'].dropna(subset=['aspect', 'sentiment'])
    assert len(df), '평가 행이 없습니다. 템플릿의 예시 행은 제외됩니다.'
    bad = df[~df['aspect'].isin(ASPECTS) | ~df['sentiment'].isin(SENTS)]
    assert bad.empty, f'속성/감성 값이 목록에 없는 행이 있습니다: {bad.head().to_dict("records")}'
    annotators = list(dict.fromkeys(df['annotator'].astype(str)))
    ids = list(dict.fromkeys(df['review_id'].astype(str)))
    text = df.drop_duplicates('review_id').set_index(df.drop_duplicates('review_id')['review_id'].astype(str))['text']

    def gold_of(ann):
        sub = df[df['annotator'].astype(str) == ann]
        return [{r.aspect: r.sentiment for r in sub[sub['review_id'].astype(str) == i].itertuples()} for i in ids]

    gold = gold_of(annotators[0])
    pred = extract(f'human_eval_{len(ids)}', [text[i] for i in ids])
    rep = dict(전체=compare(gold, pred), 속성별=per_aspect(gold, pred))
    if len(annotators) > 1:
        rep['라벨러간_일치'] = compare(gold, gold_of(annotators[1]))
    dis = [dict(review_id=i, text=text[i], 사람=json.dumps(g, ensure_ascii=False), 모델=json.dumps(p, ensure_ascii=False))
           for i, g, p in zip(ids, gold, pred) if g != p]
    return rep, pd.DataFrame(dis)
