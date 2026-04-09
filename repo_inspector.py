#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

EXCLUDED_DIRS = {
    '.git', '.hg', '.svn', '.next', '.nuxt', '.turbo', '.venv', 'venv', 'env',
    '__pycache__', 'node_modules', 'dist', 'build', 'coverage', 'target',
    'vendor', '.idea', '.vscode', '.pytest_cache', '.mypy_cache'
}

LOW_SIGNAL_PATH_PARTS = {
    'test', 'tests', '__tests__', 'spec', 'specs', 'fixture', 'fixtures',
    'example', 'examples', 'storybook', '__mocks__', 'sample', 'samples',
    'training', 'fixtures', 'datasets', 'dataset', 'corpus', 'benchmarks'
}

DOC_LIKE_EXTENSIONS = {'.md', '.txt'}

SOURCE_EXTENSIONS = {
    '.py', '.js', '.jsx', '.ts', '.tsx', '.mts', '.cts', '.java', '.kt', '.go',
    '.rs', '.rb', '.php', '.cs', '.swift', '.scala', '.sh', '.bash', '.zsh',
    '.sql', '.graphql', '.gql', '.proto', '.c', '.cc', '.cpp', '.h', '.hpp',
    '.xml', '.html', '.htm'
}

DATA_EXTENSIONS = {'.jsonl', '.csv', '.tsv'}

PROMPT_HINTS = {
    'prompt', 'prompts', 'instruction', 'instructions', 'policy', 'policies',
    'guardrail', 'guardrails', 'system', 'template'
}

TEXT_EXTENSIONS = {
    '.py', '.js', '.jsx', '.ts', '.tsx', '.mts', '.cts', '.java', '.kt', '.go', '.rs', '.rb',
    '.php', '.cs', '.swift', '.scala', '.sh', '.bash', '.zsh', '.yml', '.yaml', '.json',
    '.toml', '.ini', '.cfg', '.conf', '.env', '.md', '.txt', '.sql', '.graphql', '.gql',
    '.proto', '.xml', '.properties', '.c', '.cc', '.cpp', '.h', '.hpp'
}

HIGH_SIGNAL_FILENAMES = {
    'package.json', 'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', 'requirements.txt',
    'pyproject.toml', 'Pipfile', 'poetry.lock', 'Dockerfile', 'docker-compose.yml',
    'docker-compose.yaml', 'go.mod', 'Cargo.toml', 'Gemfile', 'pom.xml', 'build.gradle',
    'build.gradle.kts', 'settings.gradle', 'settings.gradle.kts', 'Makefile', '.env', '.env.example'
}

PATH_KEYWORDS = {
    'entrypoint': ['app', 'main', 'server', 'index', 'run', 'manage', 'cli', 'cmd'],
    'http': ['route', 'routes', 'api', 'controller', 'endpoint', 'handler', 'view', 'resolver'],
    'auth': ['auth', 'login', 'session', 'jwt', 'oauth', 'middleware', 'permission', 'guard'],
    'ai': ['openai', 'anthropic', 'llm', 'prompt', 'rag', 'embedding', 'langchain', 'agent', 'model'],
    'files': ['upload', 'file', 'storage', 'fs', 'document', 'blob'],
    'db': ['db', 'database', 'model', 'models', 'schema', 'repository', 'prisma', 'mongo', 'migration'],
    'config': ['config', 'settings', '.env', 'secrets'],
}

CONTENT_PATTERNS = {
    'entrypoint': [
        re.compile(r'\bFastAPI\s*\('),
        re.compile(r'\bFlask\s*\('),
        re.compile(r'\bstreamlit\b', re.I),
        re.compile(r'\bexpress\s*\('),
        re.compile(r'\bkoa\s*\('),
        re.compile(r'\bNestFactory\.create\b'),
        re.compile(r'\bSpringApplication\.run\b'),
        re.compile(r'\bapp\.(get|post|put|delete|use)\b'),
        re.compile(r"if __name__ == ['\"]__main__['\"]"),
        re.compile(r'\bfunc main\s*\('),
        re.compile(r'\bpublic static void main\s*\('),
    ],
    'http': [
        re.compile(r'@(app|router)\.(get|post|put|delete|patch)'),
        re.compile(r'\b(router|app)\.(get|post|put|delete|patch|use)\b'),
        re.compile(r'\b(APIRouter|Blueprint|Router)\b'),
        re.compile(r'\b(request|req|ctx)\.(json|args|form|files|body|query|params)\b'),
        re.compile(r'\bst\.(text_input|text_area|file_uploader|chat_input|form|button)\b'),
        re.compile(r'@(Controller|Get|Post|Put|Delete|Patch)\b'),
        re.compile(r'\b(http\.HandleFunc|gin\.Default|echo\.New|mux\.NewRouter)\b'),
        re.compile(r'\b(graphql|mutation|query resolver|resolver)\b', re.I),
    ],
    'auth': [
        re.compile(r'\b(jwt|oauth|auth|authorize|authentication|authorization|passport|session|guard|acl|rbac)\b', re.I),
        re.compile(r'\b(Bearer|token|access[_-]?token|refresh[_-]?token|cookie|csrf|saml)\b', re.I),
        re.compile(r'\b(middleware|interceptor|filter)\b', re.I),
    ],
    'validation': [
        re.compile(r'\b(pydantic|zod|joi|marshmallow|cerberus|class-validator|validator|ajv|yup)\b', re.I),
        re.compile(r'\b(validate|sanitize|schema|safeParse)\b', re.I),
    ],
    'external': [
        re.compile(r'\b(requests|httpx|aiohttp|axios|fetch|urllib|http\.client|got|superagent)\b'),
        re.compile(r'\b(OpenAI|Anthropic|Bedrock|LangChain|Pinecone|Supabase|boto3|stripe|twilio|slack|discord|firebase)\b'),
        re.compile(r'https?://'),
    ],
    'ai': [
        re.compile(r'\b(openai|anthropic|llm|chatcompletion|responses\.create|completions\.create|messages\.create|prompt|system_prompt|user_prompt)\b', re.I),
        re.compile(r'\b(langchain|llamaindex|embedding|rag|vector|retriever|rerank)\b', re.I),
        re.compile(r'\btemperature\b', re.I),
    ],
    'files': [
        re.compile(r'\b(open\(|Path\(|pathlib|fs\.|multer|upload|shutil|os\.remove|unlink|writeFile|readFile|tempfile)\b'),
        re.compile(r'\b(request\.files|UploadFile|FormData|multipart|IFormFile|MultipartFile)\b'),
    ],
    'subprocess': [
        re.compile(r'\b(subprocess|os\.system|exec\(|spawn\(|popen|Runtime\.getRuntime|child_process)\b')
    ],
    'db': [
        re.compile(r'\b(sqlalchemy|prisma|mongoose|sequelize|psycopg|sqlite3|redis|mongodb|postgres|typeorm|jdbc|gorm|diesel)\b', re.I),
    ],
    'secrets': [
        re.compile(r"(?i)(api[_-]?key|secret|token|password|client_secret)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
        re.compile(r'-----BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY-----'),
        re.compile(r'\b(sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b'),
    ],
}


@dataclass
class FileInfo:
    path: Path
    rel_path: str
    content: str
    score: int = 0
    tags: set[str] = field(default_factory=set)
    reasons: list[str] = field(default_factory=list)


def looks_like_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {'http', 'https', 'ssh'} or value.endswith('.git')


def clone_repo(repo_url: str) -> tuple[Path, Path]:
    parent = Path(tempfile.mkdtemp(prefix='repo-risk-'))
    repo_dir = parent / 'repo'
    result = subprocess.run(
        ['git', 'clone', '--depth', '1', repo_url, str(repo_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        # On Windows, files with characters invalid in NTFS paths (e.g. `"`) cause
        # checkout to fail even though the git objects were fully downloaded.
        # Git prints "Clone succeeded, but checkout failed." in this case — treat
        # it as a non-fatal warning and continue with whatever was checked out.
        if 'Clone succeeded, but checkout failed' in stderr and repo_dir.exists():
            print('Warning: partial checkout — retrying with invalid-path files skipped...', file=sys.stderr)
            # Force-checkout everything that can be checked out, skipping bad paths.
            subprocess.run(
                ['git', '-C', str(repo_dir), 'restore', '--source=HEAD', ':/'],
                capture_output=True,
                text=True,
            )
        else:
            message = stderr or result.stdout.strip() or 'unknown git clone error'
            raise RuntimeError(f'Unable to clone repository: {message}')
    return repo_dir, parent


def is_text_file(path: Path) -> bool:
    if path.name.lower() == 'readme.md':
        return True
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    return path.name in HIGH_SIGNAL_FILENAMES or path.name.startswith('.env')


def read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b'\x00' in data:
        return None
    for encoding in ('utf-8', 'latin-1'):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def iter_repo_files(root: Path, max_bytes: int) -> Iterable[tuple[Path, str]]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in EXCLUDED_DIRS]
        current = Path(dirpath)
        for filename in filenames:
            path = current / filename
            if not is_text_file(path):
                continue
            try:
                if path.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            content = read_text(path)
            if content is None:
                continue
            yield path, content


def is_low_signal_path(rel_path: str) -> bool:
    parts = {part.lower() for part in Path(rel_path).parts}
    return bool(parts & LOW_SIGNAL_PATH_PARTS)


def is_doc_like_file(path: Path) -> bool:
    return path.suffix.lower() in DOC_LIKE_EXTENSIONS


def is_source_like_file(path: Path) -> bool:
    return path.suffix.lower() in SOURCE_EXTENSIONS


def is_prompt_like_path(path: Path) -> bool:
    values = [path.stem.lower(), path.name.lower(), *(part.lower() for part in path.parts[-3:])]
    return any(hint in value for hint in PROMPT_HINTS for value in values)


def is_data_like_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    if parts & {'data', 'dataset', 'datasets', 'training', 'corpus', 'fixtures', 'samples', 'benchmarks'}:
        return True
    return path.suffix.lower() in DATA_EXTENSIONS


def is_manifest_file(path: Path) -> bool:
    return path.name in HIGH_SIGNAL_FILENAMES


def path_depth(rel_path: str) -> int:
    return max(0, len(Path(rel_path).parts) - 1)


def path_has_keyword(path: Path, keyword: str, tag: str) -> bool:
    name = path.name.lower()
    stem = path.stem.lower()
    parent_parts = [part.lower() for part in path.parts[-3:-1]]

    if tag == 'entrypoint':
        return keyword == stem or stem.startswith(f'{keyword}_') or stem.endswith(f'_{keyword}')

    searchable = [name, stem, *parent_parts]
    return any(keyword in value for value in searchable)


def is_readme_path(rel_path: str) -> bool:
    return Path(rel_path).name.lower().startswith('readme')


def score_file(root: Path, path: Path, content: str) -> FileInfo:
    rel_path = path.relative_to(root).as_posix()
    info = FileInfo(path=path, rel_path=rel_path, content=content)
    doc_like = is_doc_like_file(path) and not is_prompt_like_path(path)
    data_like = is_data_like_path(path) and not is_prompt_like_path(path)

    for tag, keywords in PATH_KEYWORDS.items():
        if any(path_has_keyword(path.relative_to(root), keyword, tag) for keyword in keywords):
            info.tags.add(tag)
            info.score += 4
            info.reasons.append(f'path suggests {tag}')

    if is_manifest_file(path):
        info.tags.add('config')
        info.score += 5
        info.reasons.append('high-signal project configuration')

    if path.name.lower() == 'readme.md':
        info.score += 2

    if is_source_like_file(path):
        info.score += 2
        info.reasons.append('source-like file')

    if path_depth(rel_path) == 0 and (is_source_like_file(path) or is_manifest_file(path)):
        info.score += 3
        info.reasons.append('top-level executable/config file')
    elif path_depth(rel_path) == 1 and is_source_like_file(path):
        info.score += 1
        info.reasons.append('shallow source file')

    for tag, patterns in CONTENT_PATTERNS.items():
        if doc_like and tag != 'secrets':
            continue
        if data_like and tag not in {'secrets', 'ai'}:
            continue
        matches = sum(len(pattern.findall(content)) for pattern in patterns)
        if matches:
            info.tags.add(tag)
            info.score += min(15, matches * 3)
            info.reasons.append(f'{matches} {tag} signal(s)')

    lowered_content = content.lower()
    if 'system prompt' in lowered_content or 'user prompt' in lowered_content:
        info.tags.add('ai')
        info.score += 6
        info.reasons.append('explicit prompt construction')

    if re.search(r'\b(import|from|require)\b', content):
        info.score += 1

    if is_low_signal_path(rel_path):
        info.score -= 5
        info.reasons.append('low-signal test/example path')

    if data_like:
        info.score -= 8
        info.reasons.append('data-like file deprioritized')

    if doc_like:
        info.score -= 4
        info.reasons.append('doc-like file deprioritized')

    if doc_like and len(content.strip()) < 200 and len(content.splitlines()) <= 3 and not is_manifest_file(path):
        info.score -= 6
        info.reasons.append('small text artifact deprioritized')

    return info


def load_repository(root: Path, max_bytes: int) -> list[FileInfo]:
    return [score_file(root, path, content) for path, content in iter_repo_files(root, max_bytes=max_bytes)]


def is_seed_candidate(info: FileInfo) -> bool:
    # The selector starts from files that are likely to represent executable
    # behavior, integration boundaries, or project configuration. This keeps
    # the extracted pack small while avoiding README-heavy or dataset-heavy
    # outputs that are less useful for downstream review.
    rel = Path(info.rel_path)
    if is_readme_path(info.rel_path):
        return False
    if info.score <= 0:
        return False
    if is_data_like_path(rel) and not is_prompt_like_path(rel):
        return False
    if is_doc_like_file(rel) and not (is_manifest_file(rel) or is_prompt_like_path(rel)):
        return False
    return True


def candidate_rank(info: FileInfo) -> tuple[int, int, int, str]:
    rel = Path(info.rel_path)
    seed_bonus = int(bool({'entrypoint', 'http', 'external', 'ai', 'auth', 'files', 'db', 'subprocess', 'secrets'} & info.tags))
    top_level_bonus = int(path_depth(info.rel_path) == 0)
    return (-seed_bonus, -top_level_bonus, -info.score, info.rel_path)


def add_candidate(selected: list[FileInfo], seen: set[str], candidate: FileInfo, excluded_paths: set[str]) -> bool:
    if candidate.rel_path in seen or candidate.rel_path in excluded_paths or is_readme_path(candidate.rel_path):
        return False
    if candidate.score <= 0:
        return False
    selected.append(candidate)
    seen.add(candidate.rel_path)
    return True


def select_files(infos: list[FileInfo], max_files: int, excluded_paths: set[str] | None = None) -> list[FileInfo]:
    excluded_paths = excluded_paths or set()
    candidates = [info for info in infos if is_seed_candidate(info)]

    by_tag: dict[str, list[FileInfo]] = defaultdict(list)
    for info in candidates:
        for tag in info.tags:
            by_tag[tag].append(info)

    for files in by_tag.values():
        files.sort(key=candidate_rank)

    selected: list[FileInfo] = []
    seen: set[str] = set()
    coverage_order = ['entrypoint', 'http', 'external', 'ai', 'auth', 'files', 'db', 'subprocess', 'secrets', 'validation', 'config']

    # Include a few manifests early because they often explain dependency,
    # runtime, and deployment shape even when the main execution path lives
    # elsewhere in the tree.
    manifest_candidates = sorted(
        [info for info in candidates if is_manifest_file(Path(info.rel_path))],
        key=candidate_rank,
    )
    for candidate in manifest_candidates[:3]:
        if len(selected) >= max_files:
            return selected[:max_files]
        add_candidate(selected, seen, candidate, excluded_paths)

    for tag in coverage_order:
        for candidate in by_tag.get(tag, []):
            if add_candidate(selected, seen, candidate, excluded_paths):
                break

    # After selecting seed files, expand locally around the strongest anchors.
    # This tends to pull in nearby helpers, configs, and adjacent modules
    # without needing a language-specific call graph.
    anchor_dirs = {
        str(Path(info.rel_path).parent)
        for info in selected
        if {'entrypoint', 'http', 'external', 'ai', 'auth', 'files', 'db'} & info.tags
    }
    neighbor_candidates = sorted(
        [
            info for info in candidates
            if str(Path(info.rel_path).parent) in anchor_dirs
            and Path(info.rel_path).parent != Path('.')
        ],
        key=candidate_rank,
    )
    for candidate in neighbor_candidates:
        if len(selected) >= max_files:
            break
        add_candidate(selected, seen, candidate, excluded_paths)

    ranked = sorted(candidates, key=candidate_rank)
    for candidate in ranked:
        if len(selected) >= max_files:
            break
        add_candidate(selected, seen, candidate, excluded_paths)

    return selected[:max_files]


def detect_repo_name(root: Path, source: str) -> str:
    if looks_like_url(source):
        parsed_path = urlparse(source).path.rstrip('/')
        if parsed_path.endswith('.git'):
            parsed_path = parsed_path[:-4]
        return parsed_path.split('/')[-1] or root.name
    return root.name


def read_root_readme(root: Path) -> str:
    direct = root / 'README.md'
    if direct.exists():
        return read_text(direct) or ''
    for candidate in sorted(root.glob('README*')):
        content = read_text(candidate)
        if content is not None:
            return content
    return 'README.md not found.'


def summarize_purpose(readme: str, infos: list[FileInfo]) -> str:
    lines = [re.sub(r'^#+\s*', '', line.strip()) for line in readme.splitlines() if line.strip()]
    if lines:
        description = ' '.join(lines[:3])
        if len(description) > 320:
            description = description[:317] + '...'
        return description

    tags = Counter(tag for info in infos for tag in info.tags)
    parts = []
    if tags['http']:
        parts.append('exposes application endpoints')
    if tags['ai']:
        parts.append('integrates LLM or prompt-driven behavior')
    if tags['db']:
        parts.append('persists application data')
    if tags['files']:
        parts.append('handles filesystem or upload flows')
    if not parts:
        parts.append('contains executable or integration-heavy code that should be reviewed for trust boundaries')
    return 'This repository ' + ', '.join(parts[:3]) + '.'


def infer_architecture(selected: list[FileInfo], infos: list[FileInfo]) -> list[str]:
    tags = Counter(tag for info in selected for tag in info.tags)
    lowered_paths = [info.rel_path.lower() for info in infos]
    bullets = []
    if any('/frontend/' in f or '/client/' in f or f.startswith('frontend/') or f.startswith('client/') for f in lowered_paths):
        bullets.append('Repository likely separates client-facing code from backend or integration code.')
    if tags['http']:
        bullets.append('Application exposes request-handling code with route/controller or resolver-style entry points.')
    if tags['auth'] or tags['validation']:
        bullets.append('Security logic appears distributed across auth and validation-related modules rather than isolated in one boundary.')
    if tags['ai']:
        bullets.append('AI integration is present, including prompt/model call surfaces that should be reviewed for guardrails, prompt injection resistance, and fallback behavior.')
    if tags['external'] or tags['db']:
        bullets.append('Core flows depend on external services such as APIs, model providers, databases, or cloud resources.')
    if not bullets:
        bullets.append('Architecture could not be strongly inferred; selected files focus on the highest-signal executable and integration code.')
    return bullets[:4]


# These signatures are intentionally evidence-based rather than exhaustive.
# They aim to recognize common architectural shapes across web apps, agents,
# orchestration stacks, and hosted model backends without assuming any one
# framework or provider is always present.
FRAMEWORK_PATTERNS: list[tuple[str, list[re.Pattern[str]]]] = [
    ('Flask', [re.compile(r'\bfrom flask import\b'), re.compile(r'\bFlask\s*\(__name__\)')]),
    ('FastAPI', [re.compile(r'\bfrom fastapi import\b'), re.compile(r'\bFastAPI\s*\(')]),
    ('Django', [re.compile(r'\bfrom django\b'), re.compile(r'\bdjango\.urls\b')]),
    ('Streamlit', [re.compile(r'\bimport streamlit as st\b'), re.compile(r'\bst\.(text_input|text_area|file_uploader|button)\b')]),
    ('Express', [re.compile(r'\brequire\([\'"]express[\'"]\)'), re.compile(r'\bexpress\s*\(')]),
    ('Next.js', [re.compile(r'\bnext\b', re.I), re.compile(r'\bapp/api\b', re.I)]),
    ('LangChain', [re.compile(r'\blangchain\b', re.I)]),
    ('LlamaIndex', [re.compile(r'\bllamaindex\b', re.I), re.compile(r'\bllama_index\b', re.I)]),
    ('AutoGen', [re.compile(r'\bautogen\b', re.I)]),
    ('CrewAI', [re.compile(r'\bcrewai\b', re.I)]),
]


AUTH_PATTERNS = [
    re.compile(r'\bflask_login\b', re.I),
    re.compile(r'\bLoginManager\b'),
    re.compile(r'\blogin_required\b'),
    re.compile(r'\bcurrent_user\b'),
    re.compile(r'\bJWT\b'),
    re.compile(r'\bjwt\b'),
    re.compile(r'\bAuthlib\b'),
    re.compile(r'\boauth2?\b', re.I),
    re.compile(r'\bpassport\b', re.I),
    re.compile(r'\bauthorize\b', re.I),
    re.compile(r'\b@requires_auth\b'),
    re.compile(r'\brbac\b', re.I),
    re.compile(r'\bacl\b', re.I),
    re.compile(r'\bsessionmiddleware\b', re.I),
    re.compile(r'\bnextauth\b', re.I),
]


UPLOAD_PATTERNS = [
    re.compile(r'\brequest\.files\b', re.I),
    re.compile(r'\bst\.file_uploader\b', re.I),
    re.compile(r'\bUploadFile\b'),
    re.compile(r'\bmultipart\b', re.I),
    re.compile(r'\bsecure_filename\b'),
    re.compile(r'\bupload\b', re.I),
    re.compile(r'\bDocument\('),
    re.compile(r'\bfitz\.open\b'),
]


PROVIDER_SIGNATURES: list[tuple[str, list[re.Pattern[str]]]] = [
    ('OpenAI API', [
        re.compile(r'\bfrom openai import\b'),
        re.compile(r'\bOpenAI\s*\('),
        re.compile(r'\bOPENAI_API_KEY\b'),
        re.compile(r'api\.openai\.com', re.I),
    ]),
    ('Anthropic API', [
        re.compile(r'\bfrom anthropic import\b'),
        re.compile(r'\bAnthropic\s*\('),
        re.compile(r'\bANTHROPIC_API_KEY\b'),
        re.compile(r'api\.anthropic\.com', re.I),
        re.compile(r'\bclaude-[\w.-]+\b', re.I),
    ]),
    ('Google Gemini API', [
        re.compile(r'generativelanguage\.googleapis\.com', re.I),
        re.compile(r'\bx-goog-api-key\b', re.I),
        re.compile(r'\bGEMINI_API_KEY\b'),
        re.compile(r'\bgoogle\.genai\b', re.I),
        re.compile(r'\bgenerativeai\b', re.I),
        re.compile(r'\bgemini-[\w.-]+\b', re.I),
    ]),
    ('Google Vertex AI', [
        re.compile(r'\bvertexai\b', re.I),
        re.compile(r'\baiplatform\b', re.I),
        re.compile(r'\bPROJECT_ID\b'),
        re.compile(r'vertexai\.googleapis\.com', re.I),
        re.compile(r'\bpublishers/google/models\b', re.I),
    ]),
    ('Azure OpenAI', [
        re.compile(r'openai\.azure\.com', re.I),
        re.compile(r'\bAZURE_OPENAI_', re.I),
        re.compile(r'\bAzureOpenAI\b'),
        re.compile(r'\bazure\.ai\.openai\b', re.I),
    ]),
    ('AWS Bedrock', [
        re.compile(r'\bbedrock\b', re.I),
        re.compile(r'\bbedrock-runtime\b', re.I),
        re.compile(r'\bboto3\b'),
        re.compile(r'amazonaws\.com', re.I),
        re.compile(r'\bamazon\.nova\b', re.I),
        re.compile(r'\btitan[-\w.]*\b', re.I),
    ]),
    ('Cohere API', [
        re.compile(r'\bcohere\b', re.I),
        re.compile(r'api\.cohere\.(ai|com)', re.I),
        re.compile(r'\bCOHERE_API_KEY\b'),
        re.compile(r'\bcommand-r(?:-plus)?\b', re.I),
    ]),
    ('Mistral API', [
        re.compile(r'\bmistralai\b', re.I),
        re.compile(r'api\.mistral\.ai', re.I),
        re.compile(r'\bMISTRAL_API_KEY\b'),
        re.compile(r'\b(mistral|mixtral|ministral|codestral|pixtral|devstral)[-\w.]*\b', re.I),
    ]),
    ('Together AI', [
        re.compile(r'api\.together\.ai', re.I),
        re.compile(r'\bTOGETHER_API_KEY\b'),
        re.compile(r'\btogether\b', re.I),
    ]),
    ('Groq API', [
        re.compile(r'api\.groq\.com/openai', re.I),
        re.compile(r'\bGROQ_API_KEY\b'),
        re.compile(r'\bgroq\b', re.I),
    ]),
    ('Fireworks AI', [
        re.compile(r'api\.fireworks\.ai', re.I),
        re.compile(r'\bFIREWORKS_API_KEY\b'),
        re.compile(r'\bfireworks\b', re.I),
    ]),
    ('DeepSeek API', [
        re.compile(r'api\.deepseek\.com', re.I),
        re.compile(r'\bDEEPSEEK_API_KEY\b'),
        re.compile(r'\bdeepseek-[\w.-]+\b', re.I),
    ]),
    ('Perplexity API', [
        re.compile(r'api\.perplexity\.ai', re.I),
        re.compile(r'\bPERPLEXITY_API_KEY\b'),
        re.compile(r'\bsonar(?:-[\w.]+)?\b', re.I),
        re.compile(r'\bperplexity\b', re.I),
    ]),
    ('xAI API', [
        re.compile(r'api\.x\.ai', re.I),
        re.compile(r'\bXAI_API_KEY\b'),
        re.compile(r'\bgrok-[\w.-]+\b', re.I),
        re.compile(r'\bxai\b', re.I),
    ]),
    ('OpenRouter', [
        re.compile(r'openrouter\.ai/api', re.I),
        re.compile(r'\bOPENROUTER_API_KEY\b'),
        re.compile(r'\bOpenRouter\b'),
    ]),
    ('Ollama', [
        re.compile(r'localhost:11434', re.I),
        re.compile(r'\bollama\b', re.I),
        re.compile(r'\bOLLAMA_HOST\b'),
    ]),
    ('Replicate', [
        re.compile(r'\breplicate\b', re.I),
        re.compile(r'api\.replicate\.com', re.I),
        re.compile(r'\bREPLICATE_API_TOKEN\b'),
    ]),
    ('Hugging Face Inference', [
        re.compile(r'\bhuggingface_hub\b', re.I),
        re.compile(r'\bInferenceClient\b'),
        re.compile(r'api-inference\.huggingface\.co', re.I),
        re.compile(r'\bHF_TOKEN\b'),
    ]),
    ('Cloudflare Workers AI', [
        re.compile(r'\bWorkers AI\b', re.I),
        re.compile(r'api\.cloudflare\.com/client/v4/accounts/.*/ai', re.I),
        re.compile(r'\bCLOUDFLARE_API_TOKEN\b'),
    ]),
    ('Databricks Model Serving', [
        re.compile(r'\bdatabricks\b', re.I),
        re.compile(r'/serving-endpoints/', re.I),
        re.compile(r'\bDATABRICKS_TOKEN\b'),
    ]),
    ('NVIDIA NIM', [
        re.compile(r'\bnim\b', re.I),
        re.compile(r'api\.nim\.nvidia\.com', re.I),
        re.compile(r'\bNVIDIA_API_KEY\b'),
    ]),
    ('Dify API', [
        re.compile(r'\bapi\.dify\.ai\b', re.I),
        re.compile(r'\bDify\b', re.I),
    ]),
    ('LangChain stack', [re.compile(r'\blangchain\b', re.I)]),
    ('LlamaIndex stack', [re.compile(r'\bllamaindex\b', re.I), re.compile(r'\bllama_index\b', re.I)]),
]


def combined_selected_content(selected: list[FileInfo]) -> str:
    return '\n'.join(info.content for info in selected)


def detect_framework(selected: list[FileInfo]) -> str | None:
    combined = combined_selected_content(selected[:8])
    for label, patterns in FRAMEWORK_PATTERNS:
        if any(pattern.search(combined) for pattern in patterns):
            return label
    return None


def detect_providers(selected: list[FileInfo], limit: int = 4) -> list[str]:
    combined = combined_selected_content(selected)
    found: list[str] = []
    # Provider detection accepts multiple matches because modern repositories
    # often mix direct SDK usage, gateways, local runtimes, and orchestration
    # libraries in the same codebase.
    for label, patterns in PROVIDER_SIGNATURES:
        if any(pattern.search(combined) for pattern in patterns):
            found.append(label)
        if len(found) >= limit:
            break
    return found


def extract_model_names(selected: list[FileInfo], limit: int = 3) -> list[str]:
    combined = combined_selected_content(selected)
    patterns = [
        re.compile(r'model\s*=\s*["\']([^"\']{2,80})["\']'),
        re.compile(r'"model"\s*:\s*["\']([^"\']{2,80})["\']'),
    ]
    found: list[str] = []
    for pattern in patterns:
        for match in pattern.findall(combined):
            value = match.strip()
            if value and value not in found:
                found.append(value)
            if len(found) >= limit:
                return found
    return found


def describe_ai_backend(selected: list[FileInfo]) -> str | None:
    providers = detect_providers(selected)
    model_names = extract_model_names(selected)
    provider_text = ', '.join(providers[:3]) if providers else None
    if provider_text and model_names:
        return f"{', '.join(model_names)} via {provider_text}"
    if model_names:
        return ', '.join(model_names)
    if provider_text:
        return provider_text
    return None


def has_upload_surface(selected: list[FileInfo]) -> bool:
    combined = combined_selected_content(selected)
    return any(pattern.search(combined) for pattern in UPLOAD_PATTERNS)


def has_auth_layer(selected: list[FileInfo]) -> bool:
    combined = combined_selected_content(selected)
    return any(pattern.search(combined) for pattern in AUTH_PATTERNS)


def detect_execution_modes(selected: list[FileInfo]) -> list[str]:
    combined = combined_selected_content(selected)
    modes: list[str] = []
    if re.search(r'@app\.route|\b(router|app)\.(get|post|put|delete|patch)\b|\bst\.(text_input|text_area|file_uploader|button)\b', combined):
        modes.append('interactive request-handling')
    if re.search(r'if __name__ == [\'"]__main__[\'"]|\bargparse\b|\bclick\b|\btyper\b', combined):
        modes.append('CLI or script execution')
    if re.search(r'\b(subprocess|os\.system|exec\(|spawn\(|popen)\b', combined):
        modes.append('subprocess execution')
    if re.search(r'\b(queue|worker|celery|rq|background task|asyncio\.create_task)\b', combined, re.I):
        modes.append('background or worker-style execution')
    return modes[:4]


def infer_repo_specific_risks(repo_name: str, selected: list[FileInfo]) -> list[str]:
    bullets: list[str] = []
    framework = detect_framework(selected)
    ai_backend = describe_ai_backend(selected)
    upload_surface = has_upload_surface(selected)
    auth_layer = has_auth_layer(selected)
    execution_modes = detect_execution_modes(selected)

    anchor_files = ', '.join(info.rel_path for info in selected[:4])
    if framework:
        bullets.append(f'{repo_name} appears to be a {framework}-based application, with its primary execution and request-handling surface concentrated in the extracted files ({anchor_files}).')
    elif execution_modes:
        bullets.append(f'{repo_name} appears to rely on {", ".join(execution_modes)} based on the extracted files ({anchor_files}).')
    if ai_backend:
        bullets.append(f'{repo_name} sends user-controlled content to {ai_backend}, so prompt safety, output handling, provider failures, and cost/rate-limit behavior are core architectural risks.')
    if upload_surface:
        bullets.append(f'{repo_name} exposes a file upload or document-ingestion surface, which increases risk around file type validation, size limits, parsing safety, temporary-file handling, and untrusted content reaching downstream model calls.')
    if framework and upload_surface and not auth_layer:
        bullets.append(f'The extracted files do not show a clear authentication or identity layer for {repo_name}; if this surface is intended to be private, that is a material architectural risk because upload and analysis routes may be reachable without user-level access control.')
    elif not auth_layer and any('http' in info.tags or 'entrypoint' in info.tags for info in selected):
        bullets.append(f'No clear authentication framework or access-control layer is visible in the extracted files for {repo_name}, so exposed endpoints should be reviewed as potentially unauthenticated by default.')

    if not bullets:
        bullets.append(f'No highly specific repo-level architectural risk could be inferred beyond the selected executable and integration surfaces for {repo_name}.')
    return bullets[:4]


def infer_data_flow(selected: list[FileInfo]) -> list[str]:
    bullets = []
    route_files = [info.rel_path for info in selected if 'http' in info.tags or 'entrypoint' in info.tags]
    processing_files = [info.rel_path for info in selected if {'validation', 'db', 'files'} & info.tags]
    external_files = [info.rel_path for info in selected if {'external', 'ai', 'subprocess'} & info.tags]
    auth_files = [info.rel_path for info in selected if 'auth' in info.tags]

    if route_files:
        bullets.append(f"User input likely enters through {', '.join(route_files[:4])}.")
    if has_auth_layer(selected) and auth_files:
        bullets.append(f"Trust boundaries, session checks, or permission logic likely occur in {', '.join(auth_files[:4])}.")
    elif route_files:
        bullets.append('No clear authentication boundary is visible in the extracted files, so request-handling code should be reviewed as potentially public-facing.')
    if processing_files:
        bullets.append(f"Input appears to move into validation, persistence, or file handling in {', '.join(processing_files[:4])}.")
    if external_files:
        bullets.append(f"Processed data can flow to external APIs, model providers, subprocesses, or other services via {', '.join(external_files[:4])}.")
    if not bullets:
        bullets.append('Data flow could not be strongly inferred from the selected files; review selected entry points and integrations first.')
    return bullets[:4]


def infer_risk_surfaces(selected: list[FileInfo]) -> list[str]:
    mapping = {
        'http': 'Input handling points',
        'files': 'File uploads or filesystem access',
        'external': 'External API calls',
        'ai': 'LLM or prompt construction',
        'validation': 'Validation or sanitization layers',
        'subprocess': 'System command execution',
        'secrets': 'Possible hardcoded secrets',
    }
    surfaces = []
    for tag, label in mapping.items():
        files = [info.rel_path for info in selected if tag in info.tags]
        if files:
            surfaces.append(f"- {label}: {', '.join(files[:4])}")
    if has_auth_layer(selected):
        auth_files = [info.rel_path for info in selected if 'auth' in info.tags]
        if auth_files:
            surfaces.append(f"- Authentication / authorization boundaries: {', '.join(auth_files[:4])}")
    else:
        surfaces.append('- Authentication / authorization boundaries: no clear auth layer detected in extracted files')
    if not surfaces:
        surfaces.append('- High-risk surfaces were not strongly detected; selected files still represent the most likely executable boundaries.')
    return surfaces


def evidence_summary(selected: list[FileInfo]) -> list[str]:
    lines = []
    framework = detect_framework(selected)
    ai_backend = describe_ai_backend(selected)
    if framework:
        lines.append(f'- Framework evidence: {framework}')
    if ai_backend:
        lines.append(f'- Model/provider evidence: {ai_backend}')
    if has_upload_surface(selected):
        lines.append('- Upload/ingestion evidence: file upload or document ingestion detected')
    if not has_auth_layer(selected):
        lines.append('- Auth evidence: no clear authentication framework detected in extracted files')
    return lines


def selected_file_lines(selected: list[FileInfo]) -> list[str]:
    lines = []
    for info in selected:
        reasons = ', '.join(dict.fromkeys(info.reasons[:4])) or 'high-signal executable code'
        lines.append(f'- {info.rel_path} → {reasons}')
    return lines


def emit_output(repo_name: str, purpose: str, readme: str, architecture: list[str], repo_specific_risks: list[str], data_flow: list[str], risk_surfaces: list[str], selected: list[FileInfo]) -> str:
    lines: list[str] = []
    lines.append(f'REPOSITORY: {repo_name}')
    lines.append('')
    lines.append('PURPOSE:')
    lines.append(purpose)
    lines.append('')
    lines.append('----------------------------------')
    lines.append('README')
    lines.append('----------------------------------')
    lines.append('')
    lines.append('### README.md')
    lines.append(readme.rstrip())
    lines.append('')
    lines.append('----------------------------------')
    lines.append('ARCHITECTURE:')
    lines.append('----------------------------------')
    for bullet in architecture:
        lines.append(f'- {bullet}')
    lines.append('')
    lines.append('----------------------------------')
    lines.append('REPOSITORY-SPECIFIC RISKS:')
    lines.append('----------------------------------')
    for bullet in repo_specific_risks:
        lines.append(f'- {bullet}')
    evidence = evidence_summary(selected)
    if evidence:
        lines.append('')
        lines.extend(evidence)
    lines.append('')
    lines.append('----------------------------------')
    lines.append('DATA FLOW:')
    lines.append('----------------------------------')
    for bullet in data_flow:
        lines.append(f'- {bullet}')
    lines.append('')
    lines.append('----------------------------------')
    lines.append('RISK SURFACES (IMPORTANT)')
    lines.append('----------------------------------')
    lines.extend(risk_surfaces)
    lines.append('')
    lines.append('----------------------------------')
    lines.append('SELECTED FILES:')
    lines.append('----------------------------------')
    lines.extend(selected_file_lines(selected))
    lines.append('')
    lines.append('----------------------------------')
    lines.append('CODE CONTEXT')
    lines.append('----------------------------------')
    lines.append('')
    for info in selected:
        lines.append(f'### FILE: {info.rel_path}')
        lines.append(info.content.rstrip())
        lines.append('')
    return '\n'.join(lines).rstrip() + '\n'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Extract a minimal, risk-focused subset of a repository for downstream security and architecture review.')
    parser.add_argument('repo', help='Local repository path or cloneable Git URL')
    parser.add_argument('--max-files', type=int, default=12, help='Maximum number of non-README files to include')
    parser.add_argument('--max-file-bytes', type=int, default=200000, help='Skip files larger than this many bytes')
    parser.add_argument('--output', help='Write the structured output to a file instead of stdout')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cleanup_dir: Path | None = None

    try:
        if looks_like_url(args.repo):
            repo_root, cleanup_dir = clone_repo(args.repo)
        else:
            repo_root = Path(args.repo).expanduser().resolve()
            if not repo_root.is_dir():
                raise RuntimeError(f'Repository path is not a directory: {repo_root}')

        infos = load_repository(repo_root, max_bytes=args.max_file_bytes)
        if not infos:
            raise RuntimeError('No readable text files found in repository.')

        excluded_paths: set[str] = set()
        if args.output:
            try:
                output_path = Path(args.output).expanduser().resolve()
                if output_path.is_relative_to(repo_root):
                    excluded_paths.add(output_path.relative_to(repo_root).as_posix())
            except Exception:
                pass

        selected = select_files(infos, max_files=max(1, min(args.max_files, 15)), excluded_paths=excluded_paths)
        repo_name = detect_repo_name(repo_root, args.repo)
        readme = read_root_readme(repo_root)
        purpose = summarize_purpose(readme, infos)
        architecture = infer_architecture(selected, infos)
        repo_specific_risks = infer_repo_specific_risks(repo_name, selected)
        data_flow = infer_data_flow(selected)
        risk_surfaces = infer_risk_surfaces(selected)
        output = emit_output(repo_name, purpose, readme, architecture, repo_specific_risks, data_flow, risk_surfaces, selected)

        if args.output:
            Path(args.output).write_text(output, encoding='utf-8')
        else:
            sys.stdout.write(output)
        return 0
    except Exception as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1
    finally:
        if cleanup_dir and cleanup_dir.exists():
            shutil.rmtree(cleanup_dir, ignore_errors=True)


if __name__ == '__main__':
    raise SystemExit(main())

# ============================================================
# build_evidence_pack — programmatic entry point for server.py
# ============================================================

# Minimum chars of structured sections required to skip code dumps.
# Below this threshold, we fall back to the full output so the LLM
# has raw code to reason from. Above it, the structured sections
# alone provide enough signal.
STRUCTURED_MIN_CHARS = 3000


def build_evidence_pack(
    github_url: str,
    max_files: int = 12,
    max_file_bytes: int = 200000,
    include_code_dumps: bool = False,
) -> str:
    """
    Clone a GitHub repo and return a compact, risk-focused evidence pack
    as a single string suitable for feeding to an LLM.

    Parameters:
        github_url: HTTPS URL of the repository to analyze.
        max_files: Maximum number of non-README files to include.
                   Clamped to 1..15 internally to match script's own main().
        max_file_bytes: Skip files larger than this many bytes.
        include_code_dumps: When False (default), strip the CODE CONTEXT
                            section when the structured sections are
                            substantial enough on their own. When True,
                            always return the full output.

    Returns:
        A single string containing the evidence pack. Always returns
        a usable string — on failure, returns a short error-marker
        string so callers can still make a best-effort decision.
        Error returns begin with "[repo_inspector error]".

    Notes:
        - Cleans up its temp clone directory in a finally block.
        - Never raises — all exceptions are caught and converted to
          an error-marker return string.
    """
    cleanup_dir: Path | None = None
    try:
        repo_root, cleanup_dir = clone_repo(github_url)
        infos = load_repository(repo_root, max_bytes=max_file_bytes)
        if not infos:
            return "[repo_inspector error] No readable text files found in repository."

        clamped_max = max(1, min(max_files, 15))
        selected = select_files(infos, max_files=clamped_max, excluded_paths=set())
        repo_name = detect_repo_name(repo_root, github_url)
        readme = read_root_readme(repo_root)
        purpose = summarize_purpose(readme, infos)
        architecture = infer_architecture(selected, infos)
        repo_specific_risks = infer_repo_specific_risks(repo_name, selected)
        data_flow = infer_data_flow(selected)
        risk_surfaces = infer_risk_surfaces(selected)

        full_output = emit_output(
            repo_name,
            purpose,
            readme,
            architecture,
            repo_specific_risks,
            data_flow,
            risk_surfaces,
            selected,
        )

        if include_code_dumps:
            return full_output

        # Conditional stripping: if structured sections alone are
        # substantial, return just those. Otherwise fall back to the
        # full output so the LLM has code dumps to reason from.
        marker = "CODE CONTEXT"
        marker_idx = full_output.find(marker)
        if marker_idx == -1:
            return full_output

        structured_only = full_output[:marker_idx].rstrip()

        if len(structured_only) >= STRUCTURED_MIN_CHARS:
            return structured_only + "\n"

        # Structured sections too thin — keep the full output.
        return full_output

    except Exception as exc:
        return f"[repo_inspector error] {type(exc).__name__}: {exc}"
    finally:
        if cleanup_dir and cleanup_dir.exists():
            shutil.rmtree(cleanup_dir, ignore_errors=True)