class BaseService:
	def __init__(self, name: str) -> None:
		self.name = name

	def status(self) -> str:
		return f"{self.name} ready"
