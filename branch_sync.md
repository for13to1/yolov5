# 同步官方 master 分支到个人分支

## 1. 创建个人开发分支

```bash
# 确保当前在master分支
git checkout master

# 创建并切换到新分支（假设分支名为dev）
git checkout -b dev

# 推送新分支到你的GitHub仓库
git push -u origin dev
```

## 2. 设置上游仓库（只需设置一次）

```bash
# 添加原始仓库为上游remote
git remote add upstream https://github.com/ultralytics/yolov5.git

# 验证remote设置
git remote -v
```

## 3. 同步官方更新到本地master分支

```bash
# 切换到master分支
git checkout master

# 获取上游更新
git fetch upstream

# 合并上游的master分支
git merge upstream/master

# 推送到你的GitHub仓库
git push origin master
```

## 4. 将官方更新同步到个人分支

```bash
# 切换到你的个人分支
git checkout dev

# 合并master分支的更新
git merge master

# 如果有冲突需要解决，解决后提交
git add .
git commit -m "merge upstream updates"

# 推送到远程个人分支
git push origin dev
```

## 5. 后续开发流程建议

- 日常开发在dev分支进行
- 定期执行步骤3-4保持与上游同步
- 可以通过`git fetch upstream`随时查看官方更新
- 使用`git diff master..dev`查看你的分支与master的差异

**注意事项**：

1. 合并时如果出现冲突，需要手动解决冲突文件（Git会用<<<<<<标记冲突位置）
2. 建议每次合并前先提交本地修改（保证工作区干净）
3. 可以使用`git rebase master`代替`git merge master`来保持提交历史线性（但需要更熟悉Git操作）
4. 推荐定期同步（建议每周至少同步一次），避免积累过多差异导致合并困难
