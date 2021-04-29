# rlrl

## 概要

[pfnet/pfrl](https://github.com/pfnet/pfrl)を参考に強化学習のコードを書いてる．

## Install Guide


ディレクトリ直下で

```
python3 setup.py develop
```

を実行する．`/usr/local/lib/python3.8/dist-packages/`に書き込み権限が無いとエラーが出たときは，

```
ls -l /usr/local/lib/python3.8
```

で権限を確認する．自分の場合は

```
drwxrwsr-x 2 root staff
```

となっていたため，以下のコマンドでstaffにuserを追加した．

```
sudo usermod -aG staff <my-user-name>
```

アンインストールするときは
```
python3 setup.py develop -u
```

## 実行結果

### BipedalWalker-v3 + SoftActorCritic

報酬推移と獲得方策

<img src=asset/sac-BipedalWalker-v3/result.png width=30%> <img src=asset/sac-BipedalWalker-v3/epi350.gif width=30%>

