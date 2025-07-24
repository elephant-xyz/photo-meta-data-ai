# Parallel Category Processing - Usage Guide

## Overview

The AI Image Analyzer now supports parallel category processing, allowing different categories (bedroom, kitchen, exterior, etc.) to be processed simultaneously for maximum efficiency.

## New Features

- **Parallel Category Processing**: Different categories are processed simultaneously
- **Automatic Batching**: Images are batched when there are more than 10 per category
- **Configurable Workers**: Control the number of parallel category workers
- **Cost Tracking**: Real-time cost monitoring and reporting
- **Error Handling**: Robust error handling with timeout protection

## New Command Line Options

### `--parallel-categories`
Enable parallel processing of categories within each property.

### `--category-workers <number>`
Set the maximum number of parallel workers for category processing (default: 5).

## Usage Examples

### Process S3 Properties with Parallel Categories
```bash
# Process all properties with parallel categories
python src/ai_image_analysis_optimized_multi_thread.py --all-properties --parallel-categories

# Process specific property with parallel categories
python src/ai_image_analysis_optimized_multi_thread.py --property-id 30434108090030050 --parallel-categories

# Custom category workers
python src/ai_image_analysis_optimized_multi_thread.py --all-properties --parallel-categories --category-workers 8
```

### Process Local Folders with Parallel Categories
```bash
# Process local folders with parallel categories
python src/ai_image_analysis_optimized_multi_thread.py --local-folders --parallel-categories

# Custom configuration
python src/ai_image_analysis_optimized_multi_thread.py --local-folders --parallel-categories --category-workers 6 --batch-size 10
```

## How It Works

### 1. Category Discovery
The script automatically discovers category folders within each property:
- `images/30434108090030050/bedroom/`
- `images/30434108090030050/kitchen/`
- `images/30434108090030050/exterior/`
- etc.

### 2. Parallel Processing
When `--parallel-categories` is enabled:
- Up to 5 categories processed simultaneously (configurable with `--category-workers`)
- Each category runs independently
- No dependencies between categories
- 10-minute timeout per category

### 3. Batching Logic
- **≤ 10 images**: Processed in a single batch
- **> 10 images**: Automatically split into batches of 10

### 4. Sequential vs Parallel
- **Sequential (default)**: Categories processed one after another
- **Parallel (`--parallel-categories`)**: Categories processed simultaneously

## Performance Benefits

1. **Faster Processing**: Categories don't wait for each other
2. **Better Resource Utilization**: Multiple API calls in parallel
3. **Reduced Total Time**: Especially beneficial for properties with many categories
4. **Maintained Quality**: Same analysis quality with faster processing

## Configuration Examples

### High-Performance Processing
```bash
python src/ai_image_analysis_optimized_multi_thread.py --all-properties \
    --parallel-categories \
    --category-workers 8 \
    --batch-size 10 \
    --max-workers 12
```

### Conservative Processing
```bash
python src/ai_image_analysis_optimized_multi_thread.py --all-properties \
    --parallel-categories \
    --category-workers 3 \
    --batch-size 5 \
    --max-workers 6
```

### Local Processing
```bash
python src/ai_image_analysis_optimized_multi_thread.py --local-folders \
    --parallel-categories \
    --category-workers 5 \
    --batch-size 10
```

## Monitoring and Logs

### Real-time Output
```
🚀 Using parallel category processing with 5 workers
📁 Found 6 category folders for 30434108090030050: bedroom, kitchen, exterior, bathroom, closet, living_room
🚀 Processing 6 categories in parallel for 30434108090030050
✅ Completed category bedroom (Cost: $0.0234)
✅ Completed category kitchen (Cost: $0.0456)
✅ Completed category exterior (Cost: $0.0345)
...
✅ Completed all categories for 30434108090030050 (Total cost: $0.1234)
```

### Log Files
- All operations logged to `logs/ai-analyzer.log`
- Error tracking and reporting
- Cost monitoring per category

## Error Handling

- **Category Timeout**: 10-minute timeout per category
- **API Failures**: Automatic retry with exponential backoff
- **Graceful Degradation**: Continue processing other categories if one fails
- **Resource Management**: Thread-safe operations

## Best Practices

1. **Start Conservative**: Begin with `--category-workers 3` and increase as needed
2. **Monitor API Limits**: Be aware of OpenAI rate limits
3. **Check Logs**: Review logs for any issues
4. **Test First**: Test with a single property before processing all

## Troubleshooting

### Common Issues

1. **Rate Limit Exceeded**
   ```
   OpenAI API call failed: Rate limit exceeded
   ```
   Solution: Reduce `--category-workers` or add delays

2. **Timeout Errors**
   ```
   Category bedroom processing failed: TimeoutError
   ```
   Solution: Increase timeout or reduce batch size

3. **Memory Issues**
   ```
   MemoryError: Unable to allocate array
   ```
   Solution: Reduce `--category-workers` or `--batch-size`

### Performance Tuning

- **Increase Speed**: Increase `--category-workers` (be careful with rate limits)
- **Reduce Cost**: Decrease `--batch-size`
- **Improve Stability**: Reduce `--category-workers` if experiencing timeouts
- **Handle Large Properties**: Increase timeout values

## Comparison: Sequential vs Parallel

### Sequential Processing (Default)
```
Processing Category: bedroom
✅ Completed bedroom
Processing Category: kitchen  
✅ Completed kitchen
Processing Category: exterior
✅ Completed exterior
Total time: 15 minutes
```

### Parallel Processing (`--parallel-categories`)
```
🚀 Processing 3 categories in parallel
✅ Completed category bedroom (Cost: $0.0234)
✅ Completed category kitchen (Cost: $0.0456)  
✅ Completed category exterior (Cost: $0.0345)
Total time: 5 minutes
```

## Integration with Existing Workflows

The parallel category processing integrates seamlessly with existing functionality:

- **S3 Processing**: Works with `--all-properties` and `--property-id`
- **Local Processing**: Works with `--local-folders`
- **Batching**: Respects `--batch-size` settings
- **Output**: Same output structure and format
- **Schemas**: Uses same IPFS schemas and prompts

## Migration from Sequential to Parallel

To enable parallel processing on existing workflows:

1. **Add the flag**: Add `--parallel-categories` to your command
2. **Adjust workers**: Set `--category-workers` based on your needs
3. **Monitor**: Check logs and costs
4. **Optimize**: Adjust settings based on performance

Example migration:
```bash
# Before (sequential)
python src/ai_image_analysis_optimized_multi_thread.py --all-properties

# After (parallel)
python src/ai_image_analysis_optimized_multi_thread.py --all-properties --parallel-categories
``` 