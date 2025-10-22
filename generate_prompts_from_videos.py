#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate prompts.json from Multi-View_Consistency video directory
Video filename format: "The camera orbits around. {subject_noun}, the camera circles around.-{index}"
"""

import os
import json
import re
from pathlib import Path

def extract_subject_from_filename(filename):
    """
    Extract subject_noun from video filename
    
    Filename format: "The camera orbits around. {subject_noun}, the camera circles around.-{index}.mp4"
    Example: "The camera orbits around. Acropolis, the camera circles around.-0.mp4"
    """
    # Remove .mp4 extension
    name_without_ext = filename.replace('.mp4', '')
    
    # Use regex to extract subject_noun
    # Match content between "The camera orbits around. " and ", the camera circles around.-"
    pattern = r"The camera orbits around\. (.+?), the camera circles around\.-\d+$"
    match = re.match(pattern, name_without_ext)
    
    if match:
        return match.group(1).strip()
    else:
        print(f"Warning: Could not extract subject from filename: {filename}")
        return None

def extract_index_from_filename(filename):
    """
    Extract unique index from video filename using the full filename prefix
    
    Filename format: "The camera orbits around. {subject_noun}, the camera circles around.-{index}.mp4"
    Returns: The full filename without extension as unique index
    """
    # Remove .mp4 extension and use the full name as unique index
    name_without_ext = filename.replace('.mp4', '')
    return name_without_ext

def generate_prompts_from_videos(video_dir, output_json_path, output_txt_path):
    """
    Generate prompts.json file and subject list txt file from video directory
    
    Args:
        video_dir: Video directory path
        output_json_path: Output JSON file path
        output_txt_path: Output subject list txt file path
    """
    
    video_dir = Path(video_dir)
    if not video_dir.exists():
        print(f"Error: Video directory {video_dir} does not exist")
        return
    
    # Get all mp4 files
    video_files = list(video_dir.glob("*.mp4"))
    print(f"Found {len(video_files)} video files")
    
    # Store all extracted subjects
    subjects = set()
    prompts_data = []
    
    for video_file in sorted(video_files):
        filename = video_file.name
        print(f"Processing: {filename}")
        
        # Extract subject_noun and index
        subject_noun = extract_subject_from_filename(filename)
        index = extract_index_from_filename(filename)
        
        if subject_noun and index:
            # Add to subjects set
            subjects.add(subject_noun)
            
            # Generate prompt description
            prompt = f"The camera orbits around {subject_noun}, the camera circles around."
            
            # Create prompt entry
            prompt_entry = {
                "index": index,
                "prompt": prompt,
                "subject": f"a {subject_noun}",
                "subject_noun": subject_noun,
                "place": "multi_view_consistency",
                "action": "camera_orbits",
                "filepath": str(video_file.absolute())
            }
            
            prompts_data.append(prompt_entry)
            print(f"  -> Subject: {subject_noun}, Index: {index}")
        else:
            print(f"  -> Skipped due to parsing error")
    
    # Sort by index (now using filename as unique index)
    prompts_data.sort(key=lambda x: x['index'])
    
    # Save JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(prompts_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(prompts_data)} prompts to {output_json_path}")
    
    # Save subject list to txt file
    subjects_list = sorted(list(subjects))
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for subject in subjects_list:
            f.write(f"{subject}\n")
    
    print(f"Saved {len(subjects_list)} unique subjects to {output_txt_path}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total videos processed: {len(video_files)}")
    print(f"  Valid prompts generated: {len(prompts_data)}")
    print(f"  Unique subjects: {len(subjects_list)}")
    
    # Show first few subjects as examples
    print(f"\nFirst 10 subjects:")
    for i, subject in enumerate(subjects_list[:10]):
        print(f"  {i+1}. {subject}")
    
    if len(subjects_list) > 10:
        print(f"  ... and {len(subjects_list) - 10} more")

def main():
    """Main function"""
    # Set paths
    video_dir = "../data/Multi-View_Consistency"
    output_json_path = "prompts/generated_prompts.json"
    output_txt_path = "prompts/generated_subjects.txt"
    
    # Create output directories
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    
    print("Generating prompts from Multi-View_Consistency videos...")
    print(f"Video directory: {video_dir}")
    print(f"Output JSON: {output_json_path}")
    print(f"Output TXT: {output_txt_path}")
    print("-" * 50)
    
    # Generate files
    generate_prompts_from_videos(video_dir, output_json_path, output_txt_path)
    
    print("\nGeneration completed!")

if __name__ == "__main__":
    main()