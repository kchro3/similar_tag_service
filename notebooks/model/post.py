from dataclasses import dataclass
from enum import Enum
from typing import List


class PostType(Enum):
    PHOTO = 1
    REGULAR = 2
    VIDEO = 3
    PHOTOSET = 4
    LINK = 5
    NOTE = 6
    QUOTE = 7
    AUDIO = 8
    CONVERSATION = 9


@dataclass
class Post:
    post_id: int
    blog_url: str
    post_type: PostType
    lang: str
    is_reblog: bool
    tags: List[str]
    root_tags: List[str]
