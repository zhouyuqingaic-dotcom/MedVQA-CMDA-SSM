from pathlib import Path

TRAIN_IMAGES = Path(
    "/home/yuqing/Datas/VQA-Med-2021/"
    "SYSU-HCP/extracted/data/train/images"
)
TRAIN_QA = Path(
    "/home/yuqing/Datas/VQA-Med-2021/"
    "SYSU-HCP/extracted/data/train/label.txt"
)

VAL_ROOT = Path(
    "/home/yuqing/Datas/VQA-Med-2021/extracted/"
    "validation_2021/VQA-Med-2021-Tasks-1-2-NewValidationSets"
)
VAL_IMAGES = VAL_ROOT / "ImageCLEF-2021-VQA-Med-New-Validation-Images"
VAL_QA = VAL_ROOT / "VQA-Med-2021-VQAnswering-Task1-New-ValidationSet.txt"

TEST_ROOT = Path(
    "/home/yuqing/Datas/VQA-Med-2021/extracted/"
    "test_2021/Task1-VQA-2021-TestSet-w-GroundTruth"
)

# 图片使用已经解压好的 SYSU-HCP 测试图片
TEST_IMAGES = Path(
    "/home/yuqing/Datas/VQA-Med-2021/"
    "SYSU-HCP/extracted/data/test2021/images"
)

# 问题和参考答案使用官方文件
TEST_QUESTIONS = (
    TEST_ROOT / "Task1-VQA-2021-TestSet-Questions.txt"
)
TEST_REFERENCES = (
    TEST_ROOT / "Task1-VQA-2021-TestSet-ReferenceAnswers.txt"
)


SUFFIXES = {".jpg", ".jpeg", ".png"}


def image_index(root: Path):
    return {
        p.stem: p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUFFIXES
    }


def read_ids(path: Path):
    return [
        line.split("|", 1)[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


train_images = image_index(TRAIN_IMAGES)
val_images = image_index(VAL_IMAGES)
test_images = image_index(TEST_IMAGES)

train_ids = read_ids(TRAIN_QA)
val_ids = read_ids(VAL_QA)
test_ids = read_ids(TEST_QUESTIONS)
reference_ids = read_ids(TEST_REFERENCES)

print("===== Records =====")
print("Train QA       :", len(train_ids))
print("Validation QA  :", len(val_ids))
print("Test questions :", len(test_ids))
print("Test references:", len(reference_ids))

print("\n===== Images found =====")
print("Train directory:", len(train_images))
print("Validation     :", len(val_images))
print("Test           :", len(test_images))

print("\n===== Missing images =====")
print("Train:", len(set(train_ids) - set(train_images)))
print("Val  :", len(set(val_ids) - set(val_images)))
print("Test :", len(set(test_ids) - set(test_images)))

print("\n===== Annotation consistency =====")
print("Train unique IDs:", len(set(train_ids)))
print("Val unique IDs  :", len(set(val_ids)))
print("Test unique IDs :", len(set(test_ids)))
print("Test Q/GT IDs equal:", set(test_ids) == set(reference_ids))

assert len(train_ids) == 4500
assert len(val_ids) == 500
assert len(test_ids) == 500
assert len(reference_ids) == 500

assert not (set(train_ids) - set(train_images))
assert not (set(val_ids) - set(val_images))
assert not (set(test_ids) - set(test_images))
assert set(test_ids) == set(reference_ids)

missing_test = set(test_ids) - set(test_images)
extra_test = set(test_images) - set(test_ids)

print("Missing test images:", len(missing_test))
print("Extra test images  :", len(extra_test))

assert not missing_test
assert not extra_test

print("\nPASS: VQA-Med 2021 dataset is ready.")