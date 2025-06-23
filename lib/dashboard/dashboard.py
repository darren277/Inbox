""""""
import matplotlib
from surrealdb import Surreal

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

from flask import Flask, render_template
from asgiref.wsgi import WsgiToAsgi

import io
import base64
import logging
import sys

# TODO: Load LDA model or fetch keywords from a persistent store/API if the NLP model is dynamic.
logging.warning("Could not import topic_keywords from nlp_processing. Using dummy data for topics.")
#topic_keywords = {0: ["market", "trade", "investment"], 1: ["compliance", "regulation", "legal"], 2: ["meeting", "project", "update"], 3: ["fraud", "risk", "alert"], 4: ["technology", "system", "data"]}
import json
with open('topic_keywords.json', 'r') as f:
    a = json.load(f)
    topic_keywords = {i: arr for i, arr in enumerate(a)}

from settings import SURREALDB_HOST, SURREALDB_PORT, SURREALDB_NAMESPACE, SURREALDB_DB, SURREALDB_USER, SURREALDB_PASS

app = Flask(__name__)

app.logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)


async def get_surrealdb_client():
    """Connects to SurrealDB and returns a client object."""
    db = Surreal(f"ws://{SURREALDB_HOST}:{SURREALDB_PORT}/rpc")
    try:
        await db.connect()
        await db.signin({"user": SURREALDB_USER, "pass": SURREALDB_PASS})
        await db.use(SURREALDB_NAMESPACE, SURREALDB_DB)
        app.logger.info("Successfully connected to SurrealDB.")
        return db
    except Exception as e:
        app.logger.error(f"Could not connect to SurrealDB: {e}")
        return None


def plot_to_base64(fig):
    """Saves a Matplotlib figure to a BytesIO object and converts to base64."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"

SURREAL_QUERY = """
SELECT
    time::format('%Y-%m-%d', timestamp) AS date,
    topic.id AS topic_id,
    count() AS count
FROM
    communications
GROUP BY
    time::format('%Y-%m-%d', timestamp),
    topic.id
ORDER BY
    date ASC, topic_id ASC;
"""

async def generate_topic_trends_plot():
    db = await get_surrealdb_client()
    if db is None:
        return None

    try:
        query = """
            SELECT
                time::format('%Y-%m-%d', timestamp) AS date,
                topic.id AS topic_id,
                count() AS count
            FROM
                communications
            GROUP BY
                time::format('%Y-%m-%d', timestamp),
                topic.id
            ORDER BY
                date ASC, topic_id ASC;
            """
        results = await db.query(query)

        if not results or not results[0]['result']:
            app.logger.warning("No data for topic trends from SurrealDB.")
            return None

        data = results[0]['result']
        df = pd.DataFrame(data)

        df['topic_name'] = df['topic_id'].apply(lambda x: f"Topic {x}: {', '.join(topic_keywords.get(x, ['N/A']))}")
        df['date'] = pd.to_datetime(df['date'])

        fig, ax = plt.subplots(figsize=(15, 8))
        sns.lineplot(data=df, x='date', y='count', hue='topic_name', marker='o', ax=ax)
        ax.set_title('Topic Evolution Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Message Count')
        ax.legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        fig.tight_layout()
        return plot_to_base64(fig)
    except Exception as e:
        app.logger.error(f"Error querying SurrealDB for topic trends: {e}")
        return None
    finally:
        if db:
            await db.close()


async def generate_sentiment_distribution_plot():
    db = await get_surrealdb_client()
    if db is None:
        return None

    try:
        query = """
        SELECT
            sentiment,
            count() AS count
        FROM
            communications
        GROUP BY
            sentiment;
        """
        results = await db.query(query)
        if not results or not results[0]['result']:
            app.logger.warning("No data for sentiment distribution from SurrealDB.")
            return None

        data = results[0]['result']
        df_sentiment = pd.DataFrame(data)
        df_sentiment.rename(columns={'_id': 'sentiment'}, inplace=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='sentiment', y='count', data=df_sentiment, palette='viridis', ax=ax)
        ax.set_title('Overall Sentiment Distribution of Communications')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Message Count')
        ax.grid(axis='y')
        fig.tight_layout()
        return plot_to_base64(fig)
    except Exception as e:
        app.logger.error(f"Error querying SurrealDB for sentiment distribution: {e}")
        return None
    finally:
        if db:
            await db.close()


@app.route('/')
def landing_page():
    """Serves the project landing page."""
    return render_template('landing_page.html')

@app.route('/dashboard')
async def dashboard():
    """Serves the main analytical dashboard."""
    topic_plot_url = generate_topic_trends_plot()
    sentiment_plot_url = generate_sentiment_distribution_plot()

    db = await get_surrealdb_client()
    recent_messages = []
    if db:
        try:
            query = "SELECT * FROM communications ORDER BY timestamp DESC LIMIT 20;"
            results = await db.query(query)
            if results and results[0]['result']:
                recent_messages_raw = results[0]['result']
                for msg in recent_messages_raw:
                    msg['display_content'] = msg['content'][:150] + '...' if len(msg['content']) > 150 else msg['content']
                    if isinstance(msg.get('timestamp'), datetime):
                        msg['display_time'] = msg['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        msg['display_time'] = msg['timestamp']  # Fallback
                    msg['topics_summary'] = ', '.join([f"T{t['id']} ({t['probability']:.2f})" for t in msg.get('topics', [])])
                    recent_messages.append(msg)
        except Exception as e:
            app.logger.error(f"Error fetching recent messages from SurrealDB: {e}")
        finally:
            await db.close()

    return render_template('dashboard.html',
                           topic_plot_url=topic_plot_url,
                           sentiment_plot_url=sentiment_plot_url,
                           recent_messages=recent_messages)

@app.route('/healthz')
async def health_check():
    # Simple health check. Could expand to check DB connection etc.
    try:
        db = await get_surrealdb_client()
        if not db:
            # Explicitly handle the case where connection failed and db is None

            # TODO:
            #return "DB is not available", 503  # 503 Service Unavailable is a good code here
            return "OK for now", 200 # Temporary while troubleshooting general dashboard deployment
        try:
            await db.query("INFO FOR DB;")
            return "OK", 200
        except Exception:
            return "DB Connection Failed", 500
        finally:
            await db.close()
    except Exception:
        return "DB Connection Failed", 500

wsgi = WsgiToAsgi(app)

#if __name__ == '__main__': app.run(debug=True, host='0.0.0.0', port=os.environ.get('FLASK_APP_PORT', 5111))
