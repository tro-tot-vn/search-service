import numpy as np
from services import EmbeddingService

def to_unit(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr if norm == 0 else arr / norm

def cosine(a: list[float], b: list[float]) -> float:
    ua, ub = to_unit(a), to_unit(b)
    return float(ua @ ub)  # dot của 2 vector đã chuẩn hóa = cosine

e = EmbeddingService()
r  = e.generate_dense_embedding("Phòng trọ Thủ Đức 4tr có gác, máy lạnh")
print(r)
r1 = e.generate_dense_embedding("Phòng trọ địa chỉ Thủ Đức 22m², giá 4tr, có gác lửng, máy lạnh, wc riêng, gần siêu thị")
r2 = e.generate_dense_embedding("Phòng trọ Thủ Đức 22m², giá 4tr, có gác lửng, máy lạnh, wc riêng, gần siêu thị")
r3 = e.generate_dense_embedding("Phòng trọ Quận 9 22m², giá 4tr, có gác lửng, máy lạnh, wc riêng, gần chợ")

print(cosine(r, r1))
print(cosine(r, r2))
print(cosine(r, r3))
