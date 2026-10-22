# 创建统一工具基类
from abc import ABC, abstractmethod
from typing import Any, ClassVar
from pydantic import BaseModel


class AgentTool(ABC):
    """所有Agent工具共同遵守的接口。"""

    name: ClassVar[str] # ClassVar 表明这个值是类变量，由子类定义一次，所有该子类实例共同使用，无需在创建每个实例时重复传入。
    description: ClassVar[str]

    input_model: ClassVar[
        type[BaseModel]
    ]

    def get_definition(
        self,
    ) -> dict[str, Any]:
        """生成发送给模型的工具定义。"""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema()
            }
        }

    def invoke(self, arguments: dict[str, Any],) -> dict[str, Any]:
        """验证参数、执行工具并序列化结果。"""

        validated_arguments = (
            self.input_model.model_validate(arguments)
        )

        result = self.run(validated_arguments)

        return result.model_dump(mode="json")

    @abstractmethod
    def run(
        self,
        arguments: BaseModel,
    ) -> BaseModel:
        """执行工具的具体业务逻辑。"""

        raise NotImplementedError