"""Bioresponse EDA — 1776개 익명 피처라 개별 분석 대신 집계 중심."""
import os
import logging
import matplotlib
matplotlib.use('Agg')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
OUT = os.path.join(BASE, 'EDA')
os.makedirs(OUT, exist_ok=True)

for p in ('/System/Library/Fonts/AppleSDGothicNeo.ttc',):
    if os.path.exists(p):
        matplotlib.font_manager.fontManager.addfont(p)
        plt.rcParams['font.family'] = 'Apple SD Gothic Neo'
        break
plt.rcParams['axes.unicode_minus'] = False

train = pl.read_csv(os.path.join(DATA, 'train.csv'), infer_schema_length=None)
test = pl.read_csv(os.path.join(DATA, 'test.csv'), infer_schema_length=None)
feats = [c for c in train.columns if c != 'Activity']

print('=' * 60)
print('1. 기본 정보')
print('=' * 60)
print(f'train shape : {train.shape}')
print(f'test  shape : {test.shape}')
print(f'피처 수     : {len(feats)}  (D1~D{len(feats)})')
print(f'dtype 종류  : {sorted({str(d) for d in train.dtypes})}')

# 결측
null_train = train.null_count().to_numpy()[0].sum()
null_test = test.null_count().to_numpy()[0].sum()
print(f'결측 (train/test) : {null_train} / {null_test}')

# 타깃 분포
y = train['Activity'].to_numpy()
print(f'\n타깃 Activity : 1={int(y.sum())} ({y.mean():.4f}) / 0={int((1-y).sum())} ({1-y.mean():.4f})')
print(f'all-0.5 LogLoss 기준 : 0.6931')
base_p = y.mean()
print(f'전부 train 평균({base_p:.4f}) 제출 시 LogLoss : '
      f'{-(y*np.log(base_p)+(1-y)*np.log(1-base_p)).mean():.4f}')

X = train.select(feats).to_numpy().astype(np.float64)
Xt = test.select(feats).to_numpy().astype(np.float64)

print('\n' + '=' * 60)
print('2. 피처 타입 분포')
print('=' * 60)
nuniq = np.array([len(np.unique(X[:, i])) for i in range(X.shape[1])])
const = nuniq == 1
binary = (nuniq == 2) & ~const
cont = nuniq > 2
print(f'상수 (unique=1)     : {const.sum()}')
print(f'이진 (unique=2)     : {binary.sum()}')
print(f'연속 (unique>2)     : {cont.sum()}')
print(f'  └ unique 중앙값   : {int(np.median(nuniq[cont]))}')
print(f'전체 값 범위        : [{X.min():.4f}, {X.max():.4f}]')
print(f'연속형 범위         : [{X[:, cont].min():.4f}, {X[:, cont].max():.4f}]')
if binary.sum():
    print(f'이진형 고유값       : {np.unique(X[:, binary])}')

# 이진 피처가 몰려있는 위치 확인
idx = np.arange(len(feats))
if binary.sum():
    print(f'이진 피처 인덱스 범위 : D{idx[binary].min()+1} ~ D{idx[binary].max()+1}')
if cont.sum():
    print(f'연속 피처 인덱스 범위 : D{idx[cont].min()+1} ~ D{idx[cont].max()+1}')

print('\n' + '=' * 60)
print('3. 상수 / 중복 피처')
print('=' * 60)
const_names = [feats[i] for i in idx[const]]
print(f'train 상수 피처 {len(const_names)}개 : {const_names[:20]}{" ..." if len(const_names)>20 else ""}')

# train+test 합쳐서 상수인 것도 확인 (제거 판단용)
both = np.vstack([X, Xt])
nuniq_both = np.array([len(np.unique(both[:, i])) for i in range(both.shape[1])])
const_both = [feats[i] for i in idx[nuniq_both == 1]]
print(f'train+test 모두 상수 : {len(const_both)}개')

# 중복(완전 동일 컬럼) — 해시로 1차 분류 후 정확 비교
dup_map, seen = {}, {}
for i in range(X.shape[1]):
    key = hash(X[:, i].tobytes())
    if key in seen:
        for j in seen[key]:
            if np.array_equal(X[:, i], X[:, j]):
                dup_map[feats[i]] = feats[j]
                break
        else:
            seen[key].append(i)
    else:
        seen[key] = [i]
print(f'중복 피처(train 기준 완전 동일) : {len(dup_map)}개')
for k, v in list(dup_map.items())[:10]:
    print(f'  {k} == {v}')

drop = sorted(set(const_names) | set(dup_map.keys()))
print(f'\n→ 제거 대상 합계 : {len(drop)}개  (남는 피처 {len(feats)-len(drop)}개)')

print('\n' + '=' * 60)
print('4. 타깃과의 상관 (Pearson)')
print('=' * 60)
Xc = X - X.mean(0)
yc = y - y.mean()
sd = X.std(0)
sd[sd == 0] = np.nan
corr = (Xc * yc[:, None]).mean(0) / (sd * y.std())
corr = np.nan_to_num(corr)
order = np.argsort(corr)
print('상위 15 (양의 상관):')
for i in order[::-1][:15]:
    print(f'  {feats[i]:>8}  {corr[i]:+.4f}   unique={nuniq[i]}')
print('하위 15 (음의 상관):')
for i in order[:15]:
    print(f'  {feats[i]:>8}  {corr[i]:+.4f}   unique={nuniq[i]}')
print(f'\n|corr| 최대 {np.abs(corr).max():.4f} / 평균 {np.abs(corr).mean():.4f}')
print(f'|corr| > 0.1 인 피처 : {(np.abs(corr) > 0.1).sum()}개')
print(f'|corr| > 0.2 인 피처 : {(np.abs(corr) > 0.2).sum()}개')

print('\n' + '=' * 60)
print('5. train vs test 분포 차이 (평균 기준 상위)')
print('=' * 60)
gap = np.abs(X.mean(0) - Xt.mean(0)) / (X.std(0) + 1e-9)
for i in np.argsort(gap)[::-1][:10]:
    print(f'  {feats[i]:>8}  표준화 평균차 {gap[i]:.3f}  (train {X[:,i].mean():.4f} / test {Xt[:,i].mean():.4f})')

# ---- 차트 ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(['0 (비활성)', '1 (활성)'], [int((1-y).sum()), int(y.sum())], color=['#888', '#2d7'])
ax[0].set_title(f'타깃 분포 (활성 비율 {y.mean():.3f})')
ax[1].hist(nuniq, bins=60, color='#47a')
ax[1].set_yscale('log')
ax[1].set_xlabel('피처별 고유값 개수'); ax[1].set_ylabel('피처 수 (log)')
ax[1].set_title('피처 고유값 분포 — 이진/연속 구분')
plt.tight_layout(); plt.savefig(f'{OUT}/basic_01_target_and_types.png', dpi=110); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(corr, bins=80, color='#a55')
ax[0].set_title('타깃 상관계수 분포'); ax[0].set_xlabel('corr(feature, Activity)')
top = order[::-1][:10]
ax[1].barh([feats[i] for i in top][::-1], [corr[i] for i in top][::-1], color='#2d7')
ax[1].set_title('타깃 상관 상위 10')
plt.tight_layout(); plt.savefig(f'{OUT}/basic_02_target_corr.png', dpi=110); plt.close()

# 상관 상위 6개 피처의 클래스별 분포
top6 = order[::-1][:6]
fig, axes = plt.subplots(2, 3, figsize=(13, 6))
for a, i in zip(axes.ravel(), top6):
    a.hist(X[y == 0, i], bins=40, alpha=.6, label='0', color='#888', density=True)
    a.hist(X[y == 1, i], bins=40, alpha=.6, label='1', color='#2d7', density=True)
    a.set_title(f'{feats[i]} (corr {corr[i]:+.3f})'); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f'{OUT}/basic_03_top6_by_class.png', dpi=110); plt.close()

# 이진 피처 희소도
if binary.sum():
    rate = X[:, binary].mean(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rate, bins=60, color='#75a')
    ax.set_yscale('log')
    ax.set_xlabel('이진 피처의 1 비율'); ax.set_ylabel('피처 수 (log)')
    ax.set_title(f'이진 피처 희소도 ({binary.sum()}개)')
    plt.tight_layout(); plt.savefig(f'{OUT}/basic_04_binary_sparsity.png', dpi=110); plt.close()
    print(f'\n이진 피처 1 비율 : 중앙값 {np.median(rate):.4f} / '
          f'1% 미만인 피처 {(rate < 0.01).sum()}개')

print(f'\n차트 저장 → {OUT}')
