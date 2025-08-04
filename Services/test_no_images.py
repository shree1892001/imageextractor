#!/usr/bin/env python3
"""
Test script that runs the analysis without saving images to avoid Kaleido browser issues
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Services.JDVsCV_Enhanced import EnhancedCVJDMatcher
from Common.constants import API_KEY

async def test_without_images():
    """Test the analysis without saving images."""
    
    print("🧪 Testing Analysis Without Image Saving")
    print("=" * 50)
    
    # Configuration
    config = {
        "cv_path": "E:\\imageextractor\\imageextractor\\React_JS_5_Years.pdf",
        "jd_path": "E:\\imageextractor\\imageextractor\\React_Resume.pdf",
        "enabled_modules": [
            "basic_matching",
            "culture_fit",
            "career_trajectory",
            "market_intelligence"
        ],
        "analysis_params": {
            "company_values": "Innovation, Collaboration, Continuous Learning",
            "industry": "Technology",
            "role": "Software Engineer",
            "location": "Remote",
            "save_images": False  # Disable image saving
        }
    }
    
    # Check if files exist
    if not os.path.exists(config["cv_path"]):
        print(f"❌ CV file not found: {config['cv_path']}")
        return
    
    if not os.path.exists(config["jd_path"]):
        print(f"❌ JD file not found: {config['jd_path']}")
        return
    
    print(f"✅ CV file found: {config['cv_path']}")
    print(f"✅ JD file found: {config['jd_path']}")
    
    # Create matcher instance
    matcher = EnhancedCVJDMatcher(
        api_key=API_KEY,
        enabled_modules=config["enabled_modules"]
    )
    
    try:
        # Run analysis
        print(f"\n🚀 Starting analysis...")
        results = await matcher.run_comprehensive_analysis(
            cv_path=config["cv_path"],
            jd_path=config["jd_path"],
            **config["analysis_params"]
        )
        
        print(f"\n📊 Analysis Results:")
        for module_name, result in results.items():
            print(f"  {module_name}: {type(result)}")
            if isinstance(result, dict):
                print(f"    - Keys: {list(result.keys())}")
                if "match_score" in result:
                    print(f"    - Match Score: {result['match_score']}")
                if "culture_fit_score" in result:
                    print(f"    - Culture Score: {result['culture_fit_score']}")
                if "career_growth_potential" in result:
                    print(f"    - Career Score: {result['career_growth_potential']}")
                if "error" in result:
                    print(f"    - Error: {result['error']}")
            else:
                print(f"    - Value: {str(result)[:100]}...")
        
        # Test visualization without saving images
        print(f"\n📈 Testing visualization (no image saving):")
        try:
            dashboard = matcher.create_visual_dashboard(results)
            print("✅ Dashboard created successfully")
            
            for chart_name, fig in dashboard.items():
                print(f"  📊 {chart_name}: {len(fig.data)} traces")
                if len(fig.data) > 0:
                    if hasattr(fig.data[0], 'x'):
                        print(f"    - Data points: {len(fig.data[0].x)}")
                    if hasattr(fig.data[0], 'y'):
                        print(f"    - Values: {fig.data[0].y}")
                    if hasattr(fig.data[0], 'r'):
                        print(f"    - Radar values: {fig.data[0].r}")
            
            # Display dashboard without saving
            print(f"\n📊 Displaying dashboard (no image saving):")
            matcher.display_visual_dashboard(results, save_path=None)
            
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
        
        # Generate HTML report
        print(f"\n📄 Generating HTML report:")
        try:
            report_path = matcher.generate_html_report(results)
            print(f"✅ HTML report generated: {report_path}")
        except Exception as e:
            print(f"❌ HTML report failed: {e}")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_without_images()) 