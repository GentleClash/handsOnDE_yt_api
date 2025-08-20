import json
import streamlit as st
import pyarrow as pa
import pandas as pd
import psycopg as pg
import plotly.express as px
from datetime import datetime
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="YouTube Analytics Dashboard",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #464853;
    }
    .css-1d391kg {
        background-color: #1e2129;
    }
    div[data-testid="metric-container"] {
        background-color: #262730;
        border: 1px solid #464853;
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    .stSelectbox > div > div {
        background-color: #262730;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Database connection configuration
@st.cache_resource
def init_connection() -> pg.Connection | None:
    """Initialize database connection"""
    try:
        with open("db_config.json") as f:
            custom_db_config = json.load(f)

        conn = pg.connect(
            host=custom_db_config["host"],
            port=custom_db_config["port"],
            dbname=custom_db_config["dbname"],
            user=custom_db_config["user"],
            password=custom_db_config["password"]
        )
        return conn
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(_conn) -> Tuple[pd.DataFrame, pd.DataFrame] | Tuple[None, None]:
    """Load data from database"""
    if _conn is None:
        return None, None
    
    try:
        # Load videos data
        videos_query = """
        SELECT id, title, published_at, channel_id, channel_title, 
               category_id, duration, view_count, like_count, comment_count
        FROM videos
        """
        videos_df = pd.read_sql(videos_query, _conn)
        
        # Load trending events data
        trending_query = """
        SELECT te.video_id, te.trending_date, te.first_appearance_date, 
               te.view_count as trending_view_count,
               v.title, v.channel_title, v.category_id, v.published_at,
               v.view_count, v.like_count, v.comment_count
        FROM trending_events te
        LEFT JOIN videos v ON te.video_id = v.id
        """
        trending_df = pd.read_sql(trending_query, _conn)
        
        return videos_df, trending_df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None, None

def format_number(num) -> str:
    """Format large numbers for display"""
    if pd.isna(num):
        return "N/A"
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return str(int(num))

def create_category_mapping() -> Dict[int, str]: #Not everything is covered
    """Create YouTube category ID mapping"""
    return {
        1: 'Film & Animation',
        2: 'Autos & Vehicles',
        10: 'Music',
        15: 'Pets & Animals',
        17: 'Sports',
        18: 'Short Movies',
        19: 'Travel & Events',
        20: 'Gaming',
        21: "Videoblogging",
        22: 'People & Blogs',
        23: 'Comedy',
        24: 'Entertainment',
        25: 'News & Politics',
        26: 'Howto & Style',
        27: 'Education',
        28: 'Science & Technology',
        30: "Movies",
        31: "Anime/Animation",
        34: "Comedy",
        35: "Documentary",
        44: "Trailers"
    }

def set_data_types(videos_df: pd.DataFrame, trending_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Set data types for DataFrames
    Args:
        trending_df (pd.DataFrame): DataFrame containing trending video data
        videos_df (pd.DataFrame): DataFrame containing video metadata

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing the updated DataFrames trending_df and videos_df
    """
    if trending_df is not None:
        try:
            trending_df = trending_df.astype({
                'title': 'string[pyarrow]',
                'channel_title': 'string[pyarrow]',
                'view_count': 'Int64[pyarrow]',
                'like_count': 'Int64[pyarrow]',
                'comment_count': 'Int64[pyarrow]',
                'trending_view_count': 'Int64[pyarrow]'
            })
            trending_df['trending_date'] = trending_df['trending_date'].astype(pd.ArrowDtype(pa.date32()))
            trending_df['first_appearance_date'] = trending_df['first_appearance_date'].astype(pd.ArrowDtype(pa.date32()))
            trending_df['published_at'] = trending_df['published_at'].astype(pd.ArrowDtype(pa.date32()))
        except Exception as e:
            st.error(f"Failed to set data types for trending_df: {e}")
    if videos_df is not None:
        try:
            videos_df = videos_df.astype(
                {
                    "title": "string[pyarrow]",
                    "channel_title": "string[pyarrow]",
                    "channel_id": "string[pyarrow]",
                    "view_count": "Int64[pyarrow]",
                    "like_count": "Int64[pyarrow]",
                    "comment_count": "Int32[pyarrow]"
                }
            )
            videos_df['published_at'] = videos_df['published_at'].astype(pd.ArrowDtype(pa.date32()))
            videos_df['duration'] = videos_df['duration'].astype("float32[pyarrow]")
        except Exception as e:
            st.error(f"Failed to set data types for videos_df: {e}")

    return videos_df, trending_df

def main() -> None:
    # Header
    st.set_page_config(page_title="YouTube Analytics Dashboard", page_icon="📺", layout="wide")
    st.title("📺 YouTube Analytics Dashboard")
    st.markdown("### Trending Videos Analysis")
    
    # Initialize connection
    conn = init_connection()
    
    if conn is None:
        st.error("Cannot connect to database. Please check your connection settings.")
        return
    
    # Load data
    videos_df, trending_df = load_data(conn)
    
    if videos_df is None or trending_df is None:
        st.error("Failed to load data from database.")
        return
    
    if videos_df.empty:
        st.warning("No data found in the database. Please run the extraction and loading process first.")
        return
    
    videos_df, trending_df = set_data_types(videos_df, trending_df)
    
    # Add category names
    category_mapping = create_category_mapping()
    videos_df['category_name'] = videos_df['category_id'].map(category_mapping).fillna('Other')
    trending_df['category_name'] = trending_df['category_id'].map(category_mapping).fillna('Other')
    
    # Convert to categorical with 'Other' included in categories
    all_categories = list(set(list(category_mapping.values()) + ['Other']))
    videos_df['category_name'] = videos_df['category_name'].astype(pd.CategoricalDtype(categories=all_categories))
    trending_df['category_name'] = trending_df['category_name'].astype(pd.CategoricalDtype(categories=all_categories))
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Category filter
    categories = ['All'] + sorted(videos_df['category_name'].dropna().unique().tolist())
    selected_category = st.sidebar.selectbox("Select Category", categories)
    
    # Channel filter
    channels = ['All'] + sorted(videos_df['channel_title'].dropna().unique().tolist()[:50])  # Limit to top 50
    selected_channel = st.sidebar.selectbox("Select Channel", channels)
    
    # Apply filters
    filtered_videos = videos_df.copy()
    if selected_category != 'All':
        filtered_videos = filtered_videos[filtered_videos['category_name'] == selected_category]
    if selected_channel != 'All':
        filtered_videos = filtered_videos[filtered_videos['channel_title'] == selected_channel]
    
    # Key Metrics Row
    st.header("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_videos = len(filtered_videos)
        st.metric(
            label="Total Videos",
            value=format_number(total_videos),
            delta=None
        )
    
    with col2:
        total_views = filtered_videos['view_count'].sum()
        st.metric(
            label="Total Views",
            value=format_number(total_views),
            delta=None
        )
    
    with col3:
        total_likes = filtered_videos['like_count'].sum()
        st.metric(
            label="Total Likes",
            value=format_number(total_likes),
            delta=None
        )
    
    with col4:
        avg_duration = filtered_videos['duration'].mean()
        avg_duration_formatted = f"{int(avg_duration//60)}m {int(avg_duration%60)}s" if not pd.isna(avg_duration) else "N/A"
        st.metric(
            label="Avg Duration",
            value=avg_duration_formatted,
            delta=None
        )
    
    # Charts Row 1
    st.header("📈 Video Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        # Top performing videos by views
        st.subheader("🏆 Top 10 Videos by Views")
        top_videos = filtered_videos.nlargest(10, 'view_count')[['title', 'view_count', 'channel_title']]
        top_videos['title_short'] = top_videos['title'].str[:40] + '...'
        
        fig_top_videos = px.bar(
            top_videos, 
            x='view_count', 
            y='title_short',
            orientation='h',
            labels={'view_count': 'View Count', 'title_short': 'Video Title'},
            color='view_count',
            color_continuous_scale='viridis',
            template='plotly_dark'
        )
        fig_top_videos.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_top_videos, use_container_width=True)
    
    with col2:
        # Category distribution
        st.subheader("🎯 Content by Category")
        category_counts = filtered_videos['category_name'].value_counts()
        
        fig_category = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            template='plotly_dark',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_category.update_traces(textposition='inside', textinfo='percent+label')
        fig_category.update_layout(height=400)
        st.plotly_chart(fig_category, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        # Views vs Likes scatter plot
        st.subheader("💖 Views vs Likes Relationship")
        if len(filtered_videos) > 0:
            sample_df = filtered_videos.sample(min(1000, len(filtered_videos))) if len(filtered_videos) > 1000 else filtered_videos
            # Fill NaN values in comment_count with 1 to avoid plotly size errors
            sample_df = sample_df.copy()
            sample_df['comment_count'] = sample_df['comment_count'].fillna(1)
            
            fig_scatter = px.scatter(
                sample_df,
                x='view_count',
                y='like_count',
                color='category_name',
                size='comment_count',
                hover_data=['title', 'channel_title'],
                template='plotly_dark',
                labels={'view_count': 'View Count', 'like_count': 'Like Count'}
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No data available for the selected filters.")
    
    with col2:
        # Channel performance
        st.subheader("🏢 Top Channels by Total Views")
        channel_stats = filtered_videos.groupby('channel_title').agg({
            'view_count': 'sum',
            'like_count': 'sum',
            'id': 'count'
        }).rename(columns={'id': 'video_count'}).reset_index()
        
        top_channels = channel_stats.nlargest(10, 'view_count')
        
        fig_channels = px.bar(
            top_channels,
            x='view_count',
            y='channel_title',
            orientation='h',
            color='video_count',
            labels={'view_count': 'Total Views', 'channel_title': 'Channel', 'video_count': 'Video Count'},
            template='plotly_dark',
            color_continuous_scale='plasma'
        )
        fig_channels.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_channels, use_container_width=True)
    
    # Charts Row 3
    col1, col2 = st.columns(2)
    
    with col1:
        # Duration distribution
        st.subheader("⏱️ Video Duration Distribution")
        duration_bins = pd.cut(filtered_videos['duration'].dropna(), 
                             bins=[0, 180, 600, 1800, 3600, float('inf')], 
                             labels=['0-3min', '3-10min', '10-30min', '30-60min', '60min+'])
        duration_dist = duration_bins.value_counts()
        
        fig_duration = px.bar(
            x=duration_dist.index,
            y=duration_dist.values,
            labels={'x': 'Duration Range', 'y': 'Number of Videos'},
            template='plotly_dark',
            color=duration_dist.values,
            color_continuous_scale='blues'
        )
        fig_duration.update_layout(height=400)
        st.plotly_chart(fig_duration, use_container_width=True)
    
    with col2:
        # Engagement rate by category
        st.subheader("📊 Engagement Rate by Category")
        filtered_videos['engagement_rate'] = (filtered_videos['like_count'] + filtered_videos['comment_count']) / filtered_videos['view_count'] * 100
        engagement_by_category = filtered_videos.groupby('category_name')['engagement_rate'].mean().sort_values(ascending=True)
        # Drop any category with 0 engagement
        engagement_by_category = engagement_by_category[engagement_by_category > 0]

        # Convert to DataFrame for plotly
        engagement_df = engagement_by_category.reset_index()
        engagement_df.columns = ['category_name', 'engagement_rate']

        fig_engagement = px.bar(
            engagement_df,
            x='engagement_rate',
            y='category_name',
            orientation='h',
            labels={'engagement_rate': 'Engagement Rate (%)', 'category_name': 'Category'},
            template='plotly_dark',
            color='engagement_rate',
            color_continuous_scale='reds'
        )
        fig_engagement.update_layout(height=400)
        st.plotly_chart(fig_engagement, use_container_width=True)
    
    # Data Tables Section
    st.header("📋 Data Tables")
    
    # Tabs for different data views
    tab1, tab2, tab3 = st.tabs(["🎥 Videos Overview", "🔥 Trending Videos", "📈 Channel Statistics"])
    
    with tab1:
        st.subheader("Videos Data")
        display_videos = filtered_videos[['title', 'channel_title', 'category_name', 'view_count', 'like_count', 'comment_count', 'published_at']].copy()
        display_videos['view_count'] = display_videos['view_count'].apply(format_number)
        display_videos['like_count'] = display_videos['like_count'].apply(format_number)
        display_videos['comment_count'] = display_videos['comment_count'].apply(format_number)
        st.dataframe(display_videos, use_container_width=True, height=400)
    
    with tab2:
        if not trending_df.empty:
            st.subheader("Trending Events")
            display_trending = trending_df[['title', 'channel_title', 'category_name', 'trending_date', 'first_appearance_date', 'view_count']].copy()
            display_trending['view_count'] = display_trending['view_count'].apply(format_number)
            st.dataframe(display_trending, use_container_width=True, height=400)
        else:
            st.info("No trending events data available.")
    
    with tab3:
        st.subheader("Channel Performance Summary")
        channel_summary = channel_stats.sort_values('view_count', ascending=False)
        channel_summary['view_count'] = channel_summary['view_count'].apply(format_number)
        channel_summary['like_count'] = channel_summary['like_count'].apply(format_number)
        st.dataframe(channel_summary, use_container_width=True, height=400)
    
    # Footer
    st.markdown("---")
    st.markdown("### 🔄 Data Refresh")
    if st.button("Refresh Data"):
        st.cache_data.clear()
        # To Do
        # Execute the ETL pipeline then rerun
        st.rerun()
    
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

if __name__ == "__main__":
    main()
