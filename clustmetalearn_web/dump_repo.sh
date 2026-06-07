#!/bin/bash

# Название результирующего файла
OUTPUT_FILE="django_analysis.txt"

# Очищаем или создаем файл заново
echo "=== АНАЛИЗ Django-проекта ClustMetaLearn ===" > "$OUTPUT_FILE"
echo "Дата генерации: $(date)" >> "$OUTPUT_FILE"
echo "=========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "1. СТРУКТУРА ПРОЕКТА" >> "$OUTPUT_FILE"
echo "-----------------------------------------" >> "$OUTPUT_FILE"

# Функция для построения дерева каталогов без утилиты tree
function generate_tree {
    local dir="$1"
    local prefix="$2"
    
    # Получаем список файлов и папок, исключая скрытые и служебные
    local items=$(ls -a "$dir" | grep -vE '^\.$|^\.\.$|^\.git$|^\.venv$|^__pycache__$|^\.idea$|^\.vscode$|^\.ipynb_checkpoints$|^'"$OUTPUT_FILE"'$')
    
    local count=$(echo "$items" | wc -w)
    local i=0
    
    echo "$items" | while read -r item; do
        if [ -z "$item" ]; then continue; fi
        ((i++))
        
        # Определяем символы разветвления дерева
        if [ "$i" -eq "$count" ]; then
            echo "${prefix}└── ${item}" >> "$OUTPUT_FILE"
            local next_prefix="${prefix}    "
        else
            echo "${prefix}├── ${item}" >> "$OUTPUT_FILE"
            local next_prefix="${prefix}│   "
        fi
        
        # Если это директория, уходим в рекурсию
        if [ -d "$dir/$item" ]; then
            generate_tree "$dir/$item" "$next_prefix"
        fi
    done
}

# Запускаем генерацию структуры из текущей папки
echo "." >> "$OUTPUT_FILE"
generate_tree "." ""

echo "" >> "$OUTPUT_FILE"
echo "=========================================" >> "$OUTPUT_FILE"
echo "2. СОДЕРЖИМОЕ ФАЙЛОВ" >> "$OUTPUT_FILE"
echo "=========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Обходим все файлы, исключая бинарные, скрытые и служебные директории
find . -type f \
    ! -path '*/.*' \
    ! -path './.venv/*' \
    ! -path '*/__pycache__/*' \
    ! -name "$OUTPUT_FILE" \
    ! -name "*.png" \
    ! -name "*.jpg" \
    ! -name "*.jpeg" \
    ! -name "*.pdf" \
    ! -name "*.pyc" \
    ! -name "*.pkl" \
    ! -name "*.json" | sort | while read -r file; do
    
    echo "-----------------------------------------" >> "$OUTPUT_FILE"
    echo "ФАЙЛ: $file" >> "$OUTPUT_FILE"
    echo "-----------------------------------------" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    # Добавляем содержимое файла в отчет
    cat "$file" >> "$OUTPUT_FILE"
    
    echo "" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
done

echo "Анализ завершен! Результат сохранен в файл: $OUTPUT_FILE"
