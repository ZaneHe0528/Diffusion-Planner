# 断点续传：中断后重新运行即可从断点继续；已下完的文件自动跳过
download() {
    local url="$1"
    local file="${2:-$(basename "$url")}"

    if [[ -f "$file" ]]; then
        local remote_size local_size
        remote_size=$(curl -fsI "$url" | awk 'tolower($1)=="content-length:" {print $2}' | tr -d '\r')
        local_size=$(stat -c%s "$file" 2>/dev/null || echo 0)
        if [[ -n "$remote_size" && "$local_size" -eq "$remote_size" ]]; then
            echo "[skip] $file 已下载完成 (${local_size} bytes)"
            return 0
        fi
        echo "[resume] $file 已存在 (${local_size} bytes)，继续下载..."
    else
        echo "[download] $file"
    fi

    if command -v aria2c &>/dev/null; then
        aria2c -c -x 16 -s 16 --retry-wait=5 --max-tries=0 -o "$file" "$url"
    else
        wget -c --tries=0 --timeout=30 --read-timeout=60 -O "$file" "$url"
    fi
}






# maps下载
#---------------------------------------------------------------------
# nuplan-maps-v1.1.zip  ~926 MB
#
# BASE_URL="https://d1qinkmu0ju04f.cloudfront.net/public/nuplan-v1.1"
# download "${BASE_URL}/nuplan-maps-v1.1.zip"
#---------------------------------------------------------------------






# nuPlan mini Split
#---------------------------------------------------------------------
# nuplan-v1.1_mini.zip  ~7.96 GB

# AWS Asia CDN（东京 ap-northeast-1，无需登录）
# BASE_URL="https://d1qinkmu0ju04f.cloudfront.net/public/nuplan-v1.1"
# 超大文件 CloudFront 不支持，改走 S3 直连
# S3_URL="https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1"

# download "${BASE_URL}/nuplan-v1.1_mini.zip"
#---------------------------------------------------------------------






# nuPlan Train Split
#---------------------------------------------------------------------
# 合计 ~947 GB
#
 BASE_URL="https://d1qinkmu0ju04f.cloudfront.net/public/nuplan-v1.1"
 S3_URL="https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1"
#
 download "${BASE_URL}/nuplan-v1.1_train_boston.zip"       # ~35.5 GB
 download "${BASE_URL}/nuplan-v1.1_train_pittsburgh.zip"   # ~28.5 GB
 download "${BASE_URL}/nuplan-v1.1_train_singapore.zip"    # ~32.6 GB
#
 for i in {1..6}; do
#     download "${S3_URL}/nuplan-v1.1_train_vegas_${i}.zip"
 done
# vegas_1~6: ~144 / ~142 / ~141 / ~134 / ~127 / ~164 GB
#---------------------------------------------------------------------






# nuPlan Val Split
#---------------------------------------------------------------------
# nuplan-v1.1_val.zip  ~90 GB

# S3_URL="https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1"

# download "${S3_URL}/nuplan-v1.1_val.zip"

#---------------------------------------------------------------------






# nuPlan Test Split
#---------------------------------------------------------------------
# nuplan-v1.1_test.zip  ~89 GB

# S3_URL="https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1"

# download "${S3_URL}/nuplan-v1.1_test.zip"

#---------------------------------------------------------------------
