"""
Система памяти для агентов - AgentMemory.

Реализует:
- Short-term memory (текущая сессия)
- Long-term memory (персистентное хранение в БД)
- Semantic search для поиска по истории
"""

import logging
import json
import sqlite3
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

from src.config.settings import MEMORY_DATABASE_URL

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Типы памяти."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    CONTEXT = "context"
    INSIGHTS = "insights"
    REASONING = "reasoning"


@dataclass
class MemoryRecord:
    """Запись в памяти агента."""
    id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    session_id: str
    user_id: str = "default"
    relevance_score: float = 1.0
    tags: List[str] = None
    
    def __post_init__(self):
        """Инициализация после создания."""
        if self.tags is None:
            self.tags = []
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Генерация уникального ID."""
        content_str = json.dumps(self.content, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(f"{self.timestamp}{content_str}".encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        result = asdict(self)
        result['memory_type'] = self.memory_type.value
        result['timestamp'] = self.timestamp.isoformat()
        result['content'] = json.dumps(self.content, ensure_ascii=False)
        result['metadata'] = json.dumps(self.metadata, ensure_ascii=False)
        result['tags'] = json.dumps(self.tags, ensure_ascii=False)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryRecord':
        """Создание из словаря."""
        data = data.copy()
        data['memory_type'] = MemoryType(data['memory_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['content'] = json.loads(data['content'])
        data['metadata'] = json.loads(data['metadata'])
        data['tags'] = json.loads(data['tags'])
        return cls(**data)


class AgentMemory:
    """Система памяти для агентов с поддержкой short-term и long-term хранения."""
    
    def __init__(self, session_id: str = None, user_id: str = "default"):
        """
        Инициализация системы памяти.
        
        Args:
            session_id: ID сессии для кратковременной памяти
            user_id: ID пользователя
        """
        self.session_id = session_id or self._generate_session_id()
        self.user_id = user_id
        
        # Short-term память (в оперативной памяти)
        self.short_term_memory: List[MemoryRecord] = []
        
        # Настройки
        self.max_short_term_records = 50
        
        # Инициализация БД
        self._init_database()
        
        logger.info(f"AgentMemory инициализирована для сессии {self.session_id}")
    
    def _generate_session_id(self) -> str:
        """Генерация ID сессии."""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _init_database(self) -> None:
        """Инициализация базы данных для долговременной памяти."""
        try:
            # Создаем директорию если её нет
            db_path = Path(MEMORY_DATABASE_URL)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Подключение к БД
            with sqlite3.connect(MEMORY_DATABASE_URL) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS agent_memory (
                        id TEXT PRIMARY KEY,
                        memory_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        relevance_score REAL DEFAULT 1.0,
                        tags TEXT DEFAULT '[]'
                    )
                """)
                
                # Индексы для быстрого поиска
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON agent_memory(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_session ON agent_memory(session_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_user ON agent_memory(user_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON agent_memory(memory_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_relevance ON agent_memory(relevance_score)")
                
                conn.commit()
                
            logger.info("База данных памяти инициализирована")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации БД памяти: {e}")
    
    def add_memory(self, 
                   content: Dict[str, Any], 
                   memory_type: MemoryType = MemoryType.SHORT_TERM,
                   metadata: Dict[str, Any] = None,
                   tags: List[str] = None,
                   relevance_score: float = 1.0,
                   auto_save: bool = True) -> str:
        """Добавление записи в память."""
        try:
            record = MemoryRecord(
                id="",
                memory_type=memory_type,
                content=content,
                metadata=metadata or {},
                timestamp=datetime.now(),
                session_id=self.session_id,
                user_id=self.user_id,
                relevance_score=relevance_score,
                tags=tags or []
            )
            
            self.short_term_memory.append(record)
            
            # Ограничиваем размер кратковременной памяти
            if len(self.short_term_memory) > self.max_short_term_records:
                self.short_term_memory = self.short_term_memory[-self.max_short_term_records:]
            
            # Автосохранение важных записей
            if auto_save and relevance_score >= 0.7 and memory_type in [MemoryType.INSIGHTS, MemoryType.REASONING]:
                self._save_to_long_term(record)
            
            return record.id
        except Exception as e:
            logger.error(f"Ошибка добавления записи в память: {e}")
            return ""
    
    def _save_to_long_term(self, record: MemoryRecord) -> bool:
        """Сохранение в долговременную память."""
        try:
            with sqlite3.connect(MEMORY_DATABASE_URL) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO agent_memory 
                    (id, memory_type, content, metadata, timestamp, session_id, user_id, relevance_score, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.id,
                    record.memory_type.value,
                    json.dumps(record.content, ensure_ascii=False),
                    json.dumps(record.metadata, ensure_ascii=False),
                    record.timestamp.isoformat(),
                    record.session_id,
                    record.user_id,
                    record.relevance_score,
                    json.dumps(record.tags, ensure_ascii=False)
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения в долговременную память: {e}")
            return False
    
    def search_memory(self, query: str, limit: int = 10) -> List[MemoryRecord]:
        """Поиск в памяти."""
        try:
            query_lower = query.lower()
            results = []
            
            # Поиск в кратковременной памяти
            for record in self.short_term_memory:
                content_str = json.dumps(record.content, ensure_ascii=False).lower()
                if query_lower in content_str or any(query_lower in tag.lower() for tag in record.tags):
                    results.append(record)
            
            # Поиск в долговременной памяти
            try:
                with sqlite3.connect(MEMORY_DATABASE_URL) as conn:
                    cursor = conn.execute("""
                        SELECT * FROM agent_memory 
                        WHERE user_id = ? AND (content LIKE ? OR tags LIKE ?)
                        ORDER BY relevance_score DESC, timestamp DESC
                        LIMIT ?
                    """, [self.user_id, f"%{query}%", f"%{query}%", limit])
                    
                    for row in cursor.fetchall():
                        record = MemoryRecord(
                            id=row[0],
                            memory_type=MemoryType(row[1]),
                            content=json.loads(row[2]),
                            metadata=json.loads(row[3]),
                            timestamp=datetime.fromisoformat(row[4]),
                            session_id=row[5],
                            user_id=row[6],
                            relevance_score=row[7],
                            tags=json.loads(row[8])
                        )
                        results.append(record)
            except Exception as e:
                logger.warning(f"Ошибка поиска в долговременной памяти: {e}")
            
            # Сортируем и ограничиваем
            results.sort(key=lambda x: (x.relevance_score, x.timestamp), reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Ошибка поиска в памяти: {e}")
            return []
    
    def save_insight(self, insight: str, confidence: float = 1.0, tags: List[str] = None) -> str:
        """Сохранение инсайта."""
        content = {"insight": insight, "confidence": confidence}
        metadata = {"type": "automated_insight", "source": "agent_analysis"}
        
        return self.add_memory(
            content=content,
            memory_type=MemoryType.INSIGHTS,
            metadata=metadata,
            tags=tags or ["insight"],
            relevance_score=confidence,
            auto_save=True
        )
    
    def save_reasoning_chain(self, question: str, steps: List[Dict], conclusion: str) -> str:
        """Сохранение цепочки рассуждений."""
        content = {
            "question": question,
            "reasoning_steps": steps,
            "conclusion": conclusion
        }
        metadata = {"type": "reasoning_chain"}
        
        return self.add_memory(
            content=content,
            memory_type=MemoryType.REASONING,
            metadata=metadata,
            tags=["reasoning"],
            relevance_score=0.8,
            auto_save=True
        )
    
    def get_session_context(self, max_records: int = 5) -> List[MemoryRecord]:
        """Получение контекста сессии."""
        return self.short_term_memory[-max_records:]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Статистика памяти."""
        short_term_count = len(self.short_term_memory)
        long_term_count = 0
        
        try:
            with sqlite3.connect(MEMORY_DATABASE_URL) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM agent_memory WHERE user_id = ?", [self.user_id])
                long_term_count = cursor.fetchone()[0]
        except Exception:
            pass
        
        return {
            "session_id": self.session_id,
            "short_term_records": short_term_count,
            "long_term_records": long_term_count
        }


# === ГЛОБАЛЬНАЯ ПАМЯТЬ ===
_global_memory: Optional[AgentMemory] = None


def get_agent_memory(session_id: str = None, user_id: str = "default") -> AgentMemory:
    """Получение экземпляра AgentMemory."""
    global _global_memory
    
    if _global_memory is None or (session_id and _global_memory.session_id != session_id):
        _global_memory = AgentMemory(session_id=session_id, user_id=user_id)
    
    return _global_memory


def reset_memory() -> None:
    """Сброс глобальной памяти."""
    global _global_memory
    _global_memory = None


# Тестирование
if __name__ == "__main__":
    # Настройка логгирования
    logging.basicConfig(level=logging.INFO)
    
    print("🧠 Тестирование AgentMemory...")
    
    # Создание памяти
    memory = get_agent_memory()
    print(f"✅ Память создана для сессии: {memory.session_id}")
    
    # Добавление записей
    insight_id = memory.save_insight(
        "Потребительские расходы в Москве выросли на 15% по сравнению с прошлым годом",
        confidence=0.9,
        tags=["moscow", "spending", "growth"]
    )
    print(f"✅ Инсайт сохранен: {insight_id}")
    
    # Сохранение цепочки рассуждений
    reasoning_id = memory.save_reasoning_chain(
        question="Почему выросли расходы в Москве?",
        steps=[
            {"step": 1, "action": "Анализ данных", "result": "Обнаружен рост на 15%"},
            {"step": 2, "action": "Поиск корреляций", "result": "Связь с ростом доходов"},
            {"step": 3, "action": "Сравнение с другими регионами", "result": "Москва лидирует"}
        ],
        conclusion="Рост расходов обусловлен увеличением доходов населения"
    )
    print(f"✅ Рассуждение сохранено: {reasoning_id}")
    
    # Поиск в памяти
    search_results = memory.search_memory("Москва расходы", limit=5)
    print(f"✅ Поиск выполнен, найдено записей: {len(search_results)}")
    
    # Получение статистики
    stats = memory.get_memory_stats()
    print(f"✅ Статистика: {stats['short_term_records']} краткосрочных записей")
    
    print("\n🎉 AgentMemory готова к работе!") 